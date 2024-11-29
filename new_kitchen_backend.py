from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import google.generativeai as genai
from dotenv import load_dotenv
from functools import lru_cache

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Define the generation configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Safety settings for the model
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# Initialize the Generative Model
model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    safety_settings=safety_settings,
    generation_config=generation_config,
    system_instruction=(
        "You are a non-humorous veteran to cooking. Your only purpose is to give feedback, and ways to deal with food, "
        "and improve recipes based off of set ingredients. Act professional and be very precise with your steps."
    ),
)

# Cache responses for repeated queries
@lru_cache(maxsize=100)
def fetch_response(user_question):
    return model.generate_content(user_question)

@app.route('/get-answer', methods=['GET'])
def get_answer():
    user_question = request.args.get('question')
    if not user_question:
        return jsonify({"error": "Please provide a question in the query string"}), 400

    try:
        response = fetch_response(user_question)
        generated_response = response.text[:2000]  # Trim response to 2000 characters
        return jsonify({"response": generated_response}), 200
    except Exception as e:
        print(f"Error generating content: {e}")
        return jsonify({"error": "Failed to process the request"}), 500

if __name__ == '__main__':
    app.run(debug=True)
