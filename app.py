import streamlit as st
from PIL import Image
import requests
import json
from gtts import gTTS
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import os

# Set your Google Gemini API Key
API_KEY = "YOUR_API_KEY_HERE"  # Replace with your API key
genai.configure(api_key=API_KEY)


# Langchain setup: Constructing a flexible prompt template
template = "Describe the following image and provide safety insights for detected objects: {image_description}"

# Streamlit app header
st.title("🚙⚠️ Safety-Centric Image Recognition and Speech Generation")
st.markdown("""**Key Features:**  
1. **Upload an image** securely.  
2. Get a **detailed description** of the image.  
3. Perform **object detection** with bounding boxes.  
4. Hear the description as **audio output**! 🎧  
5. Receive **safety and situational insights** based on detected objects.  
""")

# File uploader to allow image upload
uploaded_file = st.file_uploader("Upload an image file (JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Step 1: Generate an image description using Google Gemini API (Google Generative AI)
    st.header("📝 Image Description:")

    # Send the request to Gemini API for generating a description
    try:
         model = genai.GenerativeModel(model_name="gemini-1.5-flash-001")
        response = model.generate_caption(image)
        description = response.text
        st.success(f"Image Description: {description}")
    except Exception as e:
        st.error(f"Error in generating image description: {e}")

 # Step 2: Audio Description
    st.header("🔊 Audio Description")
    try:
        audio_path = "description_audio.mp3"
        tts = gTTS(description, lang='en')
        tts.save(audio_path)
        st.audio(audio_path, format="audio/mp3")
    except Exception as e:
        st.error(f"Error in audio description: {e}")

st.title("📦 Object Detection")
st.markdown("Upload an image to detect objects using Generative AI.")

# File uploader
uploaded_file = st.file_uploader("Upload an image file (JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Use Generative AI to detect objects
    st.header("Detected Objects:")
    try:
        detection_prompt = """
        Analyze the uploaded image and list all objects present in it. 
        Provide the list of detected objects in plain text.
        """
        # Send the image and prompt to the Generative AI model
        model = genai.GenerativeModel(model_name="gemini-1.5-flash-001")
        response = model.generate_content(
            image=image,
            prompt=detection_prompt
        )

        detected_objects = response.text.strip()
        st.write("Detected Objects:")
        st.success(detected_objects)

    except Exception as e:
        st.error(f"Error in detecting objects: {e}")
        
    
    # Step 4: Langchain's flexible prompt handling for generating safety insights
    st.header("🛡️ Safety Insights:")
    
    # Define detected objects in the image (This would be generated based on actual object detection, here simulated)
    detected_objects = ["car", "person", "dog"]  # Example objects detected in the image
    
    # Generate safety tips for each detected object
    safety_tips = []
    
    for obj in detected_objects:
        safety_prompt = f"Provide safety tips for a detected {obj} in the image."
        # Use Langchain to generate safety insights for each object
        safety_tip = llm_chain.run(image_description=safety_prompt)
        safety_tips.append(f"- **{obj}**: {safety_tip}")

    # Display safety insights
    if safety_tips:
        st.write("The following safety tips were generated based on the detected objects:")
        for tip in safety_tips:
            st.write(tip)
    else:
        st.warning("No objects detected for safety insights.")
else:
    st.info("Please upload an image to proceed.")
