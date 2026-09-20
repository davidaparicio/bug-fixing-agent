from flask import Flask, request, jsonify
from flask_cors import CORS
from agent import run_agent
import logging
import os

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_AUTH_KEY = os.getenv("API_AUTH_KEY")


@app.route('/api/run', methods=['POST'])
def run_engine():
    if API_AUTH_KEY:
        auth = request.headers.get('Authorization')
        if not auth or auth != f"Bearer {API_AUTH_KEY}":
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

    return jsonify(result)


if __name__ == '__main__':
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug, host='0.0.0.0', port=5000)
