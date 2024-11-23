# GenAI_Final_Project

# Real-Time Scene Understanding, Object Detection, and Safety Insights 🚗

This project provides a web application that allows users to upload an image and gain insights into the scene through the following features:

- **Scene Description**: A detailed textual description of the contents of the image, generated using Google's **Gemini 1.5 flash**.
- **Object Detection & Safety Insights**: Identifies objects in the image and offers safety recommendations based on those objects.
- **Text-to-Speech**: Converts the generated scene description into speech for accessibility.

### **Features:**
- Upload an image to see the **scene description**.
- Receive **safety recommendations** based on detected objects like cars, trucks, and pedestrians.
- Listen to the description of the scene via **Text-to-Speech** conversion.

### **Tech Stack:**
- **Langchain**: Framework for interacting with Google's Generative AI model.
- **Streamlit**: Framework to create interactive web applications.
- **Google Gemini 1.5-flash**: AI model used to generate detailed descriptions of uploaded images.
- **gTTS (Google Text-to-Speech)**: Converts text descriptions into audio for accessibility.
  
