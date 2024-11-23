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
api_key = "YOUR_GOOGLE_API_KEY"

# Google Gemini endpoint
api_url = "https://gemini.googleapis.com/v1beta1/text:generate"  # Please verify this endpoint in the Google docs

# Langchain setup: Constructing a flexible prompt template
template = "Describe the following image: {image_description}"

# Langchain prompt and model setup
prompt = PromptTemplate(input_variables=["image_description"], template=template)
chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
llm_chain = LLMChain(prompt=prompt, llm=chat_model)

# Streamlit app header
st.title("🚙⚠️ Safety-Centric Image Recognition and Speech Generation with Google Gemini and Langchain")

# File uploader to allow image upload
uploaded_file = st.file_uploader("Upload an image file (JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Step 1: Generate an image description using Google Gemini API (Google Generative AI)
    st.header("📝 Image Description Generation:")

    # Send the request to Gemini API for generating a description
    try:
        prompt_description = "Describe the image content in detail."

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        payload = {
            "input": prompt_description,
            "model": "gemini-1.5-flash",  # Using the Gemini 1.5 Flash model for description generation
            "temperature": 0.7,           # Adjust the creativity of responses
            "max_tokens": 200             # Set the length of the response
        }

        response = requests.post(api_url, headers=headers, data=json.dumps(payload))

        if response.status_code == 200:
            response_data = response.json()
            description = response_data.get("output", "No description available.")
            st.success(description)

            # Step 2: Generate audio of the description using gTTS
            st.header("🔊 Audio Description:")
            audio_file_path = "description.mp3"
            tts = gTTS(description, lang='en')
            tts.save(audio_file_path)

            # Play the audio
            st.audio(audio_file_path, format="audio/mp3")

            # Cleanup temporary audio file
            os.remove(audio_file_path)

        else:
            st.error("Error occurred while generating the description.")
            st.error(response.text)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")


    # Object detection using OpenCV 
    st.header("📦 Object Detection:")

    # Convert the uploaded image to an OpenCV format
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    # Convert to grayscale (simplified object detection for demo purposes)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Example: Use Haar cascades for object detection (replace with your preferred method)
    # Load a pre-trained Haar Cascade for face detection (can be replaced with any detection model)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img_cv, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Convert back to RGB for displaying in Streamlit
    img_cv_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_cv_rgb)

    # Display the annotated image with bounding boxes
    st.image(img_pil, caption="Object Detection Results", use_container_width=True)

else:
    st.info("Please upload an image to proceed.")


