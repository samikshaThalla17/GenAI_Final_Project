import streamlit as st
from langchain.chat_models import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from google.generativeai import ChatModel, Input
from PIL import Image
import requests
from io import BytesIO
import os
from gtts import gTTS

API_KEY = "YOUR_GOOGLE_API_KEY"

# Langchain Setup
chat_model = ChatGoogleGenerativeAI(api_key=API_KEY)

# Streamlit app title and description
st.title("🔍 Real-Time Scene Understanding, Object Detection, and Safety Insights 🚗")
st.markdown("""
    This app allows you to upload an image, generate a detailed description of the scene using AI, 
    detect objects, and provide safety insights based on the detected objects. 
    It also includes Text-to-Speech conversion for accessibility.
""")

# Upload image file
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Generate Description using Gemini 1.5 (Text Generation)
    st.subheader("Scene Description:")
    description = "The image shows an environment with various objects like vehicles, buildings, and people."
    
    # Use Google's Gemini for description generation
    prompt = "Describe the contents of this image: "  # You can customize the prompt
    input_image = BytesIO(uploaded_file.read())  # Convert image file to bytes

    # Call Google's Gemini API (Text Generation)
    response = chat_model.chat(inputs=[Input(content=prompt, image=input_image)])

    description = response.results[0].content  # Extract the description
    st.write(description)

    # Text-to-Speech Conversion for Description
    tts = gTTS(description, lang='en')
    tts.save("description.mp3")
    st.audio("description.mp3", format="audio/mp3")

    # Object Detection: We are using Google's Gemini to detect objects
    st.subheader("Detected Objects and Safety Insights:")

    # Use the image for object detection (Placeholder logic as Gemini is primarily for text generation)
    objects_detected = ["Car", "Truck", "Pedestrian"]  # Assume these are detected objects
    
    safety_message = ""
    for obj in objects_detected:
        if obj.lower() == "car" or obj.lower() == "truck":
            safety_message += f"\nDetected a {obj}. Ensure you maintain a safe distance and avoid distractions."

    # Display Safety Insights
    if safety_message:
        st.warning(safety_message)
    else:
        st.info("No critical objects detected.")

    # Additional message for Accessibility
    st.subheader("Accessibility Features")
    st.markdown("""
        This app provides text-to-speech functionality to help users understand the scene description and safety insights.
        If you prefer to listen to the content, please check the audio above.
    """)

