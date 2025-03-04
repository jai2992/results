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


def encode_image(image):
    """ Convert image to base64 for API input """
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def extract_subjects_and_grades(image_base64):
    """ LLM call to extract subjects and grades from result image """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Extract subjects and grades from this result image.\n"
                        "Format the response as:\n"
                        "**Subjects & Grades:**\n"
                        "- Subject 1: Grade\n"
                        "- Subject 2: Grade\n"
                        "..."
                    )
                },
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image_base64}}
            ]
        }
    ]
    
    completion = client.chat.completions.create(
        model="meta-llama/Llama-3.2-90B-Vision-Instruct", 
        messages=messages, 
    )
    
    return completion.choices[0].message['content']

def extract_subjects_and_credits(image_base64):
    """ LLM call to extract subjects and their credit points from credits image """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Extract subjects and credit points from this credits image.\n"
                        "Format the response as:\n"
                        "**Subjects & Credits:**\n"
                        "- Subject 1: Credit Points\n"
                        "- Subject 2: Credit Points\n"
                        "..."
                    )
                },
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image_base64}}
            ]
        }
    ]
    
    completion = client.chat.completions.create(
        model="meta-llama/Llama-3.2-90B-Vision-Instruct", 
        messages=messages
    )
    
    return completion.choices[0].message['content']

def calculate_sgpa(subject_grades, subject_credits):
    """ LLM call to compute SGPA using extracted grades and credits """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        '''
                        First multiply the credit point of each subject to the grade point you have earned in the subject. Then, add all the total grade points you have got. Now, divide this sum by the total sum of credit points of the semester. You will get your SGPA.
O - 10
A+ - 9
A - 8
B+ - 7
B - 6
                        '''
                    )
                }
            ]
        }
    ]
    
    completion = client.chat.completions.create(
        model="meta-llama/Llama-3.2-90B-Vision-Instruct", 
        messages=messages
    )
    
    return completion.choices[0].message['content']

# Streamlit UI
st.title("📊 SGPA Calculator from Result & Credits Images")

st.write("📌 **Steps to Use:**")
st.write("""
1. Upload **your result image** (shows subjects & grades).
2. Upload **your credits image** (shows subjects & credit points).
3. The AI will extract and match the data.
4. It will compute **your SGPA** and show the steps.
""")

# File uploaders
result_file = st.file_uploader("📤 Upload Result Image (Subjects & Grades)", type=["jpg", "jpeg", "png"])
credits_file = st.file_uploader("📤 Upload Credits Image (Subjects & Credit Points)", type=["jpg", "jpeg", "png"])

if result_file and credits_file:
    # Display uploaded images
    result_img = Image.open(result_file)
    credits_img = Image.open(credits_file)
    
    st.image(result_img, caption="📜 Result Image", use_column_width=True)
    st.image(credits_img, caption="📝 Credits Image", use_column_width=True)

    # Convert images to base64
    result_base64 = encode_image(result_img)
    credits_base64 = encode_image(credits_img)

    # Extract subjects & grades
    st.write("⏳ **Extracting subjects & grades from result image...**")
    subjects_grades = extract_subjects_and_grades(result_base64)
    st.write("📜 **Extracted Subjects & Grades:**")
    st.code(subjects_grades, language="markdown")

    # Extract subjects & credit points
    st.write("⏳ **Extracting subjects & credit points from credits image...**")
    subjects_credits = extract_subjects_and_credits(credits_base64)
    st.write("📜 **Extracted Subjects & Credits:**")
    st.code(subjects_credits, language="markdown")

    # Compute SGPA
    st.write("⏳ **Calculating SGPA...**")
    sgpa_result = calculate_sgpa(subjects_grades, subjects_credits)

    # Display SGPA result
    st.success(f"🎯 **Final SGPA Calculation:**\n\n{sgpa_result}")
