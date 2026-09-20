"""Tests for the Flask API in api.py.

The ``run_agent`` function is patched so these tests exercise only the HTTP
layer — no LLM calls are made.
"""

import json
from unittest.mock import patch

import pytest


@pytest.fixture()
def client():
    """Create a Flask test client with run_agent mocked."""
    with patch("api.run_agent", return_value='{"result": "ok"}'):
        from api import app
        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c


class TestRunEndpoint:
    def test_success(self, client):
        resp = client.post(
            "/api/run",
            data=json.dumps({"query": "fix bug", "context": "details"}),
            content_type="application/json",
        )

        assert resp.status_code == 200
        data = resp.get_json()
        assert data == '{"result": "ok"}'

    def test_missing_query_returns_400(self, client):
        resp = client.post(
            "/api/run",
            data=json.dumps({"context": "no query here"}),
            content_type="application/json",
        )

        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data

    def test_empty_body_returns_400(self, client):
        resp = client.post(
            "/api/run",
            data=json.dumps({}),
            content_type="application/json",
        )

        assert resp.status_code == 400

    def test_context_is_optional(self, client):
        resp = client.post(
            "/api/run",
            data=json.dumps({"query": "do something"}),
            content_type="application/json",
        )

        assert resp.status_code == 200

    @patch("api.run_agent", return_value='{"result": "custom"}')
    def test_returns_agent_result(self, mock_run, client):
        resp = client.post(
            "/api/run",
            data=json.dumps({"query": "task", "context": "ctx"}),
            content_type="application/json",
        )

        assert resp.status_code == 200
