"""End-to-end tests that exercise the real smolagents -> LiteLLM -> HTTP pipeline.

These are **integration tests** — not run by default.  Use markers to opt in::

    pytest -m e2e                          # mock-server E2E (no GPU needed)
    pytest -m ollama                       # real Ollama on localhost:11434
    pytest -m lmstudio                     # real LM Studio on localhost:1234
    pytest -m "e2e or ollama or lmstudio"  # all integration tests
"""

import json
import os
import socket
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from textwrap import dedent

import pytest


# ---------------------------------------------------------------------------
# Helpers – tiny OpenAI-compatible mock server
# ---------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _chat_completion_response(content: str, model: str = "mock-model") -> dict:
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


class _MockHandler(BaseHTTPRequestHandler):
    """Handles POST /v1/chat/completions with a canned agent-terminating reply."""

    server: "_MockLLMServer"

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        self.server.requests.append(body)

        reply_text = self.server.next_reply()
        payload = json.dumps(_chat_completion_response(reply_text))

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload.encode())

    def log_message(self, *_args):
        pass


class _MockLLMServer(HTTPServer):
    """HTTPServer subclass that lets tests inspect received requests."""

    def __init__(self, port: int, replies: list[str] | None = None):
        super().__init__(("127.0.0.1", port), _MockHandler)
        self.requests: list[dict] = []
        self._replies = list(replies) if replies else []
        self._reply_idx = 0

    def next_reply(self) -> str:
        if self._reply_idx < len(self._replies):
            r = self._replies[self._reply_idx]
            self._reply_idx += 1
            return r
        return dedent("""\
            Thought: Providing the answer.
            <code>
            final_answer("mocked e2e result")
            </code>""")


@pytest.fixture()
def mock_llm_server():
    """Start a mock OpenAI-compatible server for the duration of one test."""
    port = _free_port()
    server = _MockLLMServer(port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()


@pytest.fixture()
def local_litellm_model(mock_llm_server, real_litellm_model_class):
    """Create a real LiteLLMModel that talks to the mock server."""
    port = mock_llm_server.server_address[1]
    return real_litellm_model_class(
        model_id="openai/mock-model",
        api_base=f"http://127.0.0.1:{port}/v1",
        api_key="sk-fake",
        temperature=0,
    )


# ---------------------------------------------------------------------------
# Mock-server E2E tests (run with: pytest -m e2e)
# ---------------------------------------------------------------------------

@pytest.mark.e2e
class TestE2EWithMockServer:
    """Full pipeline: real LiteLLMModel -> real HTTP -> mock server."""

    def test_litellm_model_reaches_mock(self, mock_llm_server, local_litellm_model):
        """LiteLLMModel.generate() sends a real HTTP request to the mock."""
        from smolagents import ChatMessage

        msg = ChatMessage(role="user", content=[{"type": "text", "text": "hello"}])
        result = local_litellm_model.generate([msg])

        assert len(mock_llm_server.requests) >= 1
        assert result.content is not None

    def test_code_agent_e2e(self, mock_llm_server, local_litellm_model, real_code_agent_class):
        """CodeAgent runs a task end-to-end through a real LiteLLM HTTP call."""
        from smolagents import tool

        @tool
        def greet(name: str) -> str:
            """Returns a greeting.

            Args:
                name: The name to greet.
            """
            return f"Hello, {name}!"

        agent = real_code_agent_class(tools=[greet], model=local_litellm_model)
        result = agent.run("Say hello to Alice")

        assert result == "mocked e2e result"
        assert len(mock_llm_server.requests) >= 1

    def test_mock_server_receives_correct_format(self, mock_llm_server, local_litellm_model):
        """Verify the request sent to the server is OpenAI-compatible."""
        from smolagents import ChatMessage

        msg = ChatMessage(role="user", content=[{"type": "text", "text": "test"}])
        local_litellm_model.generate([msg])

        req = mock_llm_server.requests[0]
        assert "messages" in req
        assert "model" in req
        assert any(m.get("role") == "user" for m in req["messages"])

    def test_multiple_turns(self, mock_llm_server, real_litellm_model_class, real_code_agent_class):
        """Agent handles a multi-turn conversation via the mock server."""
        from smolagents import tool

        port = mock_llm_server.server_address[1]

        # First reply: call a tool; second reply: final answer
        mock_llm_server._replies = [
            dedent("""\
                Thought: I need to greet the user first.
                <code>
                result = greet(name="World")
                print(result)
                </code>"""),
            dedent("""\
                Thought: Now I have the greeting, return it.
                <code>
                final_answer("Hello, World!")
                </code>"""),
        ]

        model = real_litellm_model_class(
            model_id="openai/mock-model",
            api_base=f"http://127.0.0.1:{port}/v1",
            api_key="sk-fake",
            temperature=0,
        )

        @tool
        def greet(name: str) -> str:
            """Returns a greeting.

            Args:
                name: The name to greet.
            """
            return f"Hello, {name}!"

        agent = real_code_agent_class(tools=[greet], model=model)
        result = agent.run("Greet the world")

        assert result == "Hello, World!"
        assert len(mock_llm_server.requests) >= 2


# ---------------------------------------------------------------------------
# Ollama E2E test (run with: pytest -m ollama)
# ---------------------------------------------------------------------------

def _ollama_available() -> bool:
    try:
        import urllib.request
        resp = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return resp.status == 200
    except Exception:
        return False


def _lmstudio_available() -> bool:
    try:
        import urllib.request
        resp = urllib.request.urlopen("http://localhost:1234/v1/models", timeout=2)
        return resp.status == 200
    except Exception:
        return False


@pytest.mark.ollama
@pytest.mark.e2e
@pytest.mark.skipif(not _ollama_available(), reason="Ollama not running on localhost:11434")
class TestE2EWithOllama:
    """Run against a real Ollama instance.

    Requires: ``ollama serve`` running and a small model pulled, e.g.::

        ollama pull smollm2:135m
    """

    def _get_first_model(self) -> str:
        import urllib.request
        resp = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5)
        data = json.loads(resp.read())
        models = data.get("models", [])
        if not models:
            pytest.skip("No Ollama models pulled — run: ollama pull smollm2:135m")
        return models[0]["name"]

    def test_ollama_litellm_roundtrip(self, real_litellm_model_class):
        """LiteLLMModel sends a real request to Ollama and gets a response."""
        from smolagents import ChatMessage

        model_name = self._get_first_model()
        model = real_litellm_model_class(
            model_id=f"ollama/{model_name}",
            api_base="http://localhost:11434",
        )
        msg = ChatMessage(role="user", content=[{"type": "text", "text": "Say hi in one word"}])
        result = model.generate([msg])
        assert result.content is not None
        assert len(result.content) > 0


@pytest.mark.lmstudio
@pytest.mark.e2e
@pytest.mark.skipif(not _lmstudio_available(), reason="LM Studio not running on localhost:1234")
class TestE2EWithLMStudio:
    """Run against a real LM Studio instance.

    Requires: LM Studio running with a model loaded on port 1234.
    """

    def test_lmstudio_litellm_roundtrip(self, real_litellm_model_class):
        """LiteLLMModel sends a real request to LM Studio and gets a response."""
        from smolagents import ChatMessage

        model = real_litellm_model_class(
            model_id="openai/local-model",
            api_base="http://localhost:1234/v1",
            api_key="lm-studio",
        )
        msg = ChatMessage(role="user", content=[{"type": "text", "text": "Say hi in one word"}])
        result = model.generate([msg])
        assert result.content is not None
        assert len(result.content) > 0
