# Code Engine for Bug Fixing

A Python-based agent that helps non-technical users fix bugs and make changes to codebases through natural language requests.

This POC was created as part of the [Dust](https://dust.tt/) AI Agents Hackathon in Paris.

## Features

- Uses LLM to analyze and fix code issues
- Git integration for pulling, committing, and pushing changes
- File management capabilities (read/write)
- REST API for integration with other applications
- Bearer token authentication (optional)
- CORS support with configurable origins
- Health check endpoint

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file based on `.env.example`:
   ```
   MODEL_ID=anthropic/claude-3-5-sonnet-latest
   API_KEY=your_api_key
   CODEBASE_PATH=/path/to/codebase
   API_AUTH_KEY=your_secret_key
   FLASK_DEBUG=false
   ALLOWED_ORIGINS=https://your-frontend.example.com
   ```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `MODEL_ID` | Yes | LiteLLM model identifier (e.g. `anthropic/claude-3-5-sonnet-latest`) |
| `API_KEY` | Yes | API key for the LLM provider |
| `CODEBASE_PATH` | Yes | Absolute path to the codebase the agent operates on |
| `API_AUTH_KEY` | No | Bearer token for API authentication. If unset, the API is unauthenticated |
| `FLASK_DEBUG` | No | Set to `true` to enable Flask debug mode (default: `false`) |
| `ALLOWED_ORIGINS` | No | Comma-separated list of allowed CORS origins. If unset, all origins are allowed |

## Usage

### As a command-line tool:
```
python agent.py "fix the button click handler in main.js"
```

### As an API server (development):
```
python api.py
```

### As an API server (production):
```
pip install gunicorn
gunicorn api:app --bind 0.0.0.0:5000 --timeout 300 --workers 2
```

Health check:
```
curl http://localhost:5000/health
```

Then send POST requests to `/api/run` with JSON body:
```bash
curl -X POST http://localhost:5000/api/run \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_secret_key" \
  -d '{"query": "fix the button click handler in main.js", "context": "additional context"}'
```

## Technical Details

- Built with smolagents `ToolCallingAgent` from HuggingFace (tool-calls only, no arbitrary code execution)
- Uses LiteLLM for model access
- Git operations handled through subprocess, relying on a user being authenticated on the system
- `python api.py` runs Flask's single-threaded dev server; use gunicorn (or another WSGI server) for production

## Next steps

In order to improve the agent's capabilities for medium and large code base, we would need to have some kind of codebase embedding and RAG.
