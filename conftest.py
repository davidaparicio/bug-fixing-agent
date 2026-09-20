"""
Root conftest: patches LiteLLMModel so that importing agent.py never hits a
real LLM provider.  The patch is function-scoped and skipped for tests
marked ``e2e``, so end-to-end tests can use the real smolagents stack.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Save real classes before anything patches them
# ---------------------------------------------------------------------------

from smolagents import LiteLLMModel as _RealLiteLLMModel
from smolagents import CodeAgent as _RealCodeAgent


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
# Function-scoped patches — skipped for tests marked "e2e"
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _mock_litellm(request):
    """Patch smolagents classes so ``import agent`` never touches LiteLLM.

    Skipped for tests marked ``e2e`` — those use the real classes.
    """
    if "e2e" in {m.name for m in request.node.iter_markers()}:
        yield
        return

    with patch("smolagents.LiteLLMModel", FakeLiteLLMModel), \
         patch("smolagents.CodeAgent", FakeCodeAgent):
        for mod_name in list(sys.modules):
            if mod_name in ("agent", "api"):
                del sys.modules[mod_name]
        yield

    for mod_name in list(sys.modules):
        if mod_name in ("agent", "api"):
            del sys.modules[mod_name]


@pytest.fixture(autouse=True)
def _set_env(monkeypatch):
    """Provide safe default env vars for every test."""
    monkeypatch.setenv("MODEL_ID", "fake-provider/fake-model")
    monkeypatch.setenv("API_KEY", "sk-fake-test-key")
    monkeypatch.setenv("CODEBASE_PATH", "/tmp/test-codebase")


# ---------------------------------------------------------------------------
# Fixtures that expose the real (unpatched) classes for E2E tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def real_litellm_model_class():
    """Return the real LiteLLMModel class, bypassing any mock."""
    return _RealLiteLLMModel


@pytest.fixture()
def real_code_agent_class():
    """Return the real CodeAgent class, bypassing any mock."""
    return _RealCodeAgent
