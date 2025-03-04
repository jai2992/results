import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image
import io
import base64

# Hugging Face API Configuration
client = InferenceClient(
    provider="fireworks-ai",
    api_key=st.secrets["huggingface"]["api_key"]  # Fetch API key from secrets.toml
)

# Rest of your code remains unchanged...
def encode_image(image):
    """ Convert image to base64 for API input """
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def extract_subjects_and_grades(image_base64):
    # ... ( unchanged ) ...

def extract_subjects_and_credits(image_base64):
    # ... ( unchanged ) ...

def calculate_sgpa(subject_grades, subject_credits):
    # ... ( unchanged ) ...

# Streamlit UI
st.title("📊 SGPA Calculator from Result & Credits Images")
# ... ( rest of the code remains unchanged ) ...
