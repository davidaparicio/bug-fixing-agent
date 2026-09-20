from flask import Flask, request, jsonify
from flask_cors import CORS
from agent import run_agent
import hmac
import logging
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024  # 1 MB

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",")
ALLOWED_ORIGINS = [o.strip() for o in ALLOWED_ORIGINS if o.strip()]
if ALLOWED_ORIGINS:
    CORS(app, origins=ALLOWED_ORIGINS)
else:
    CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_AUTH_KEY = os.getenv("API_AUTH_KEY")


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})


@app.route('/api/run', methods=['POST'])
def run_engine():
    if API_AUTH_KEY:
        auth = request.headers.get('Authorization', '')
        expected = f"Bearer {API_AUTH_KEY}"
        if not hmac.compare_digest(auth, expected):
            return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    if data is None:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    logger.info("Received request for /api/run")
    query = data.get('query')
    context = data.get('context', '')

    if not query:
        return jsonify({"error": "Missing required parameter: query"}), 400

    try:
        result = run_agent(query, context)
    except Exception:
        logger.exception("Agent execution failed")
        return jsonify({"error": "Agent execution failed"}), 500

    try:
        return jsonify(result)
    except TypeError:
        return jsonify({"result": str(result)})


if __name__ == '__main__':
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug, host='0.0.0.0', port=5000)
