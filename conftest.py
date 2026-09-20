"""
Root conftest: patches LiteLLMModel so that importing agent.py never hits a
real LLM provider.  The patch is session-scoped — it is installed once before
any test and stays active for the entire pytest run.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fake LiteLLMModel that records calls instead of reaching the network
# ---------------------------------------------------------------------------

class FakeLiteLLMModel:
    """Drop-in replacement for smolagents.LiteLLMModel.

    * Accepts the same constructor kwargs so module-level instantiation works.
    * Exposes a ``calls`` list for assertions.
    """

    def __init__(self, *args, **kwargs):
        self.model_id = kwargs.get("model_id", "fake-model")
        self.api_key = kwargs.get("api_key", "fake-key")
        self.kwargs = kwargs
        self.calls: list[dict] = []

    def __call__(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        return MagicMock(content='{"result": "mocked response"}')


# ---------------------------------------------------------------------------
# Fake CodeAgent — avoids the real orchestration loop entirely
# ---------------------------------------------------------------------------

class FakeCodeAgent:
    """Drop-in replacement for smolagents.CodeAgent.

    ``run()`` returns a canned JSON string so callers that parse the output
    still work.
    """

    def __init__(self, *args, **kwargs):
        self.tools = kwargs.get("tools", [])
        self.model = kwargs.get("model")
        self.prompt_templates = {"system_prompt": ""}
        self.run_calls: list[dict] = []

    def run(self, task, **kwargs):
        self.run_calls.append({"task": task, **kwargs})
        return '{"result": "mocked agent response"}'


# ---------------------------------------------------------------------------
# Session-scoped patches — applied before any module import of agent/api
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True, scope="session")
def _mock_litellm_globally():
    """Patch smolagents classes so ``import agent`` never touches LiteLLM."""
    with patch("smolagents.LiteLLMModel", FakeLiteLLMModel), \
         patch("smolagents.CodeAgent", FakeCodeAgent):
        # Clear cached modules so each import picks up the patch
        for mod_name in list(sys.modules):
            if mod_name in ("agent", "api"):
                del sys.modules[mod_name]
        yield
    # Cleanup: drop cached modules so other test sessions start fresh
    for mod_name in list(sys.modules):
        if mod_name in ("agent", "api"):
            del sys.modules[mod_name]


@pytest.fixture(autouse=True)
def _set_env(monkeypatch):
    """Provide safe default env vars for every test."""
    monkeypatch.setenv("MODEL_ID", "fake-provider/fake-model")
    monkeypatch.setenv("API_KEY", "sk-fake-test-key")
    monkeypatch.setenv("CODEBASE_PATH", "/tmp/test-codebase")
