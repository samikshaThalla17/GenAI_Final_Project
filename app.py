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
template = "Describe the following image and provide safety insights for detected objects: {image_description}"

# Langchain prompt and model setup
prompt = PromptTemplate(input_variables=["image_description"], template=template)
chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
llm_chain = LLMChain(prompt=prompt, llm=chat_model)

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
    
    st.header("📦 Object Detection:")
    yolo_model = YOLO("yolov8n.pt")  # Load YOLO model (Nano version for speed)
    
    # Save the uploaded image locally for YOLO processing
    uploaded_image_path = "uploaded_image.jpg"
    image.save(uploaded_image_path)

    # Perform object detection
    results = yolo_model(uploaded_image_path)
    detected_results = results[0]  # Get the first result

    # Annotated image with bounding boxes
    annotated_image = detected_results.plot()
    # Convert NumPy array (annotated_image) to Pillow Image for Streamlit
    annotated_image_pil = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))

    # Display annotated image with bounding boxes
    st.image(annotated_image_pil, caption="Object Detection Results", use_container_width=True)

    
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
