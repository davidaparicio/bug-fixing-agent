"""Tests for the tool functions defined in agent.py.

Every test mocks filesystem / subprocess so no real I/O happens and no
real LLM calls are made (LiteLLMModel is already patched in conftest.py).
"""

import os
import textwrap
from unittest.mock import MagicMock, mock_open, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_agent():
    """Import agent lazily (after session-scoped patches are active)."""
    import agent
    return agent


# ---------------------------------------------------------------------------
# get_codebase
# ---------------------------------------------------------------------------

class TestGetCodebase:
    def test_returns_file_contents(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CODEBASE_PATH", str(tmp_path))
        (tmp_path / "hello.py").write_text("print('hello')")

        agent_mod = _import_agent()
        result = agent_mod.get_codebase()

        assert "hello.py" in result
        assert "print('hello')" in result

    def test_skips_hidden_files(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CODEBASE_PATH", str(tmp_path))
        (tmp_path / ".secret").write_text("hidden")
        (tmp_path / "visible.txt").write_text("shown")

        agent_mod = _import_agent()
        result = agent_mod.get_codebase()

        assert ".secret" not in result
        assert "visible.txt" in result

    def test_skips_hidden_directories(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CODEBASE_PATH", str(tmp_path))
        hidden_dir = tmp_path / ".git"
        hidden_dir.mkdir()
        (hidden_dir / "config").write_text("git config")
        (tmp_path / "main.py").write_text("code")

        agent_mod = _import_agent()
        result = agent_mod.get_codebase()

        assert "git config" not in result
        assert "main.py" in result

    def test_empty_codebase(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CODEBASE_PATH", str(tmp_path))

        agent_mod = _import_agent()
        result = agent_mod.get_codebase()

        assert result == ""

    def test_nested_files(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CODEBASE_PATH", str(tmp_path))
        sub = tmp_path / "src"
        sub.mkdir()
        (sub / "app.py").write_text("flask app")

        agent_mod = _import_agent()
        result = agent_mod.get_codebase()

        assert "app.py" in result
        assert "flask app" in result


# ---------------------------------------------------------------------------
# pull_code
# ---------------------------------------------------------------------------

class TestPullCode:
    @patch("subprocess.check_output")
    def test_runs_git_pull(self, mock_subprocess, monkeypatch):
        monkeypatch.setenv("CODEBASE_PATH", "/repo")
        mock_subprocess.return_value = b"Already up to date.\n"

        agent_mod = _import_agent()
        result = agent_mod.pull_code()

        assert "Already up to date." in result
        mock_subprocess.assert_called_once()
        call_cmd = mock_subprocess.call_args[0][0]
        assert "git pull" in call_cmd
        assert "/repo" in call_cmd


# ---------------------------------------------------------------------------
# commit_code
# ---------------------------------------------------------------------------

class TestCommitCode:
    @patch("subprocess.check_output")
    def test_runs_git_commit(self, mock_subprocess, monkeypatch):
        monkeypatch.setenv("CODEBASE_PATH", "/repo")
        mock_subprocess.return_value = b"[main abc1234] fix typo\n"

        agent_mod = _import_agent()
        result = agent_mod.commit_code("fix typo")

        assert "fix typo" in result
        call_cmd = mock_subprocess.call_args[0][0]
        assert "git add" in call_cmd
        assert "git commit" in call_cmd


# ---------------------------------------------------------------------------
# push_code
# ---------------------------------------------------------------------------

class TestPushCode:
    @patch("subprocess.check_output")
    def test_runs_git_push(self, mock_subprocess, monkeypatch):
        monkeypatch.setenv("CODEBASE_PATH", "/repo")
        mock_subprocess.return_value = b"Everything up-to-date\n"

        agent_mod = _import_agent()
        result = agent_mod.push_code()

        assert "Everything up-to-date" in result
        call_cmd = mock_subprocess.call_args[0][0]
        assert "git push" in call_cmd


# ---------------------------------------------------------------------------
# get_file_contents
# ---------------------------------------------------------------------------

class TestGetFileContents:
    def test_reads_file(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("file content here")

        agent_mod = _import_agent()
        result = agent_mod.get_file_contents(str(f))

        assert result == "file content here"

    def test_missing_file_raises(self):
        agent_mod = _import_agent()
        with pytest.raises(FileNotFoundError):
            agent_mod.get_file_contents("/nonexistent/path.txt")


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------

class TestWriteFile:
    def test_writes_content(self, tmp_path):
        f = tmp_path / "output.txt"

        agent_mod = _import_agent()
        returned_path = agent_mod.write_file(str(f), "new content")

        assert returned_path == str(f)
        assert f.read_text() == "new content"

    def test_overwrites_existing(self, tmp_path):
        f = tmp_path / "output.txt"
        f.write_text("old")

        agent_mod = _import_agent()
        agent_mod.write_file(str(f), "new")

        assert f.read_text() == "new"


# ---------------------------------------------------------------------------
# list_all_files
# ---------------------------------------------------------------------------

class TestListAllFiles:
    def test_lists_files(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CODEBASE_PATH", str(tmp_path))
        (tmp_path / "a.py").write_text("")
        sub = tmp_path / "lib"
        sub.mkdir()
        (sub / "b.py").write_text("")

        agent_mod = _import_agent()
        result = agent_mod.list_all_files()

        paths = [os.path.basename(p) for p in result]
        assert "a.py" in paths
        assert "b.py" in paths

    def test_excludes_hidden(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CODEBASE_PATH", str(tmp_path))
        (tmp_path / ".hidden").write_text("")
        (tmp_path / "visible.py").write_text("")

        agent_mod = _import_agent()
        result = agent_mod.list_all_files()

        basenames = [os.path.basename(p) for p in result]
        assert ".hidden" not in basenames
        assert "visible.py" in basenames

    def test_empty_directory(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CODEBASE_PATH", str(tmp_path))

        agent_mod = _import_agent()
        result = agent_mod.list_all_files()

        assert result == []
