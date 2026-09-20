# Code Engine for Bug Fixing

A Python-based agent that helps non-technical users fix bugs and make changes to codebases through natural language requests.

This POC was created as part of the [Dust](https://dust.tt/) AI Agents Hackathon in Paris.

## Features

- Uses LLM to analyze and fix code issues
- Git integration for pulling, committing, and pushing changes
- File management capabilities (read/write)
- REST API for integration with other applications

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with the following variables:
   ```
   MODEL_ID=your_model_name
   API_KEY=your_api_key
   CODEBASE_PATH=/path/to/codebase
   ```

## Usage

### As a command-line tool:
```
python agent.py "fix the button click handler in main.js"
```

### As an API server:
```
python api.py
```

Then send POST requests to `/api/run` with JSON body:
```json
{
  "query": "fix the button click handler in main.js",
  "context": "additional context about the application"
}
```

## Technical Details

- Built with smolagents from HuggingFace
- Uses LiteLLM for model access
- Git operations handled through subprocess, relying on a user being authenticated on the system

## Testing

The project has a two-tier test suite that never calls paid LLM APIs.

### Install test dependencies

```
pip install -r requirements-dev.txt
```

### Unit tests (default)

```bash
pytest
```

Runs 25 fast tests with LiteLLM and CodeAgent fully mocked — no network, no GPU, no API key needed.

### E2E / integration tests

These exercise the real `smolagents → LiteLLM → HTTP` pipeline against a local server.

```bash
# Mock OpenAI-compatible server (spun up in-process, no GPU needed)
pytest -m e2e

# Real Ollama (requires: ollama serve + ollama pull smollm2:135m)
pytest -m ollama

# Real LM Studio (requires: LM Studio running on port 1234)
pytest -m lmstudio

# All integration tests at once
pytest -m "e2e or ollama or lmstudio"
```

The Ollama and LM Studio tests auto-skip when the server is not reachable.

## Next steps

In order to improve the agent's capabilities for medium and large code base, we would need to have some kind of codebase embedding and RAG.
