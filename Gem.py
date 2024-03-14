import streamlit as st
import google.generativeai as genai

# Set up the Gemini API key and model configuration
genai.configure(api_key="AIzaSyDbUQj2jSe1THDWuFVdGKRCJ7ozrzd1MyA")
generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

model = genai.GenerativeModel(
    model_name="gemini-1.0-pro",
    generation_config=generation_config,
    safety_settings=safety_settings,
)

# Streamlit app code
st.title("Gemini Chatbot")

# User input for the prompt
prompt_text = st.text_input("Enter your prompt:")

# Generate button to trigger text generation
if st.button("Generate Text"):
    # Start a conversation with Gemini and send the prompt
    convo = model.start_chat(history=[])
    convo.send_message(prompt_text)

    # Display the generated text response
    st.success("Generated Text:")
    st.write(convo.last.text)

# Add additional Streamlit components as needed for your frontend
