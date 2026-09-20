"""Tests for the run_agent function and module-level wiring.

LiteLLMModel and CodeAgent are replaced by fakes in conftest.py, so these
tests verify the orchestration without making any real LLM calls.
"""

from unittest.mock import patch


def _import_agent():
    import agent
    return agent


class TestRunAgent:
    def test_returns_mocked_response(self):
        agent_mod = _import_agent()
        result = agent_mod.run_agent("fix the typo", "some context")

        assert result == '{"result": "mocked agent response"}'

    def test_passes_query_and_context_to_agent(self):
        agent_mod = _import_agent()
        agent_mod.agent.run_calls.clear()

        agent_mod.run_agent("change color to blue", "CSS file")

        assert len(agent_mod.agent.run_calls) == 1
        task = agent_mod.agent.run_calls[0]["task"]
        assert "change color to blue" in task
        assert "CSS file" in task

    def test_system_prompt_extended(self):
        agent_mod = _import_agent()
        prompt = agent_mod.agent.prompt_templates["system_prompt"]

        assert "bug fixing agent" in prompt

    def test_model_constructed_with_env_vars(self, monkeypatch):
        """The FakeLiteLLMModel records the kwargs it was constructed with."""
        agent_mod = _import_agent()

        assert agent_mod.model.model_id is not None
        assert agent_mod.model.api_key is not None


class TestAgentTools:
    def test_agent_has_expected_tools(self):
        agent_mod = _import_agent()

        tool_names = [t.name if hasattr(t, "name") else str(t) for t in agent_mod.agent.tools]
        assert "pull_code" in tool_names
        assert "list_all_files" in tool_names
        assert "write_file" in tool_names
        assert "get_file_contents" in tool_names
