from flask import Flask, jsonify, request
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

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
        "You are a non-humorous veteran to cooking. Your only purpose is to give feedback, and ways to deal with food, and improve recipes based off of set ingredients. "
        "Anything regarding cooking is your field of work."
        "Act professional and be very precise with your steps."
        "Make your steps understandable but yet good"
    ),
)

# Initialize the chat session

# Define the GET route
@app.route('/get-answer', methods=['GET'])
def get_answer():
    # Get the user's question from the query string
    user_question = request.args.get('question')
    
    if not user_question:
        return jsonify({"error": "Please provide a question in the query string"}), 400

    # Generate a response using the chat session
    try:
        response = model.generate_content(f"{user_question}")
        generated_response = response.text  # Extract the text from the response
        return jsonify({"response": generated_response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
