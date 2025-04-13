import streamlit as st
from openai import OpenAI
import base64
import io
from PIL import Image

# Setup
OpenAI.api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI()
st.set_page_config(page_title="DALL·E 3 Image Generator with GPT", page_icon="🎨")

st.title("🎨 DALL·E 3 Image Generator (Text + Image Inputs)")

# Sidebar options
st.sidebar.header("Options")
use_gpt_assist = st.sidebar.checkbox("Use GPT-4 to assist with prompt (if image is uploaded)")

# --- Step 1: Input section ---
text_input = st.text_area("Your idea (text)", placeholder="Describe the image you want to create.", height=100)
uploaded_image = st.file_uploader("Optional reference image (if you want GPT to analyze it)", type=["jpg", "jpeg", "png"])

def encode_image(image_data):
    return base64.b64encode(image_data).decode("utf-8")

# --- Step 2: Generate prompt using GPT ---
if st.button("Generate Image"):
    if not text_input and not uploaded_image:
        st.warning("Please enter a description or upload an image.")
    else:
        with st.spinner("Generating prompt and image..."):
            try:
                # Create the base GPT-4 message
                messages = [
                    {"role": "system", "content": "You are a creative assistant that crafts prompts for DALL·E 3 image generation. If an image is provided, consider its contents to refine the prompt."},
                    {"role": "user", "content": text_input}
                ]
                
                # If an image is uploaded, include it in the prompt
                if uploaded_image:
                    image = Image.open(uploaded_image)
                    if image.mode == 'RGBA':
                        image = image.convert("RGB")
                    buffered = io.BytesIO()
                    image.save(buffered, format="JPEG")
                    image_data = buffered.getvalue()
                    base64_image = encode_image(image_data)

                    messages[1]["content"] += "\nPlease consider the following image in crafting the prompt."
                    messages.append({
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]
                    })

                # Use GPT-4 to refine the prompt
                chat_response = client.chat.completions.create(
                    model="gpt-4" if uploaded_image else "gpt-4-turbo",  # Use GPT-4 or GPT-4-turbo
                    messages=messages
                )
                refined_prompt = chat_response.choices[0].message.content.strip()

                st.success("Generated Prompt:")
                st.text_area("Prompt for DALL·E 3", refined_prompt, height=100)

                # --- Step 3: Generate image using DALL·E 3 ---
                st.info("Generating image with DALL·E 3...")
                image_response = client.images.generate(
                    model="dall-e-3",
                    prompt=refined_prompt,
                    size="1024x1024",
                    quality="standard",
                    n=1
                )
                image_url = image_response.data[0].url
                st.image(image_url, caption="Generated Image by DALL·E 3", use_container_width=True)

            except Exception as e:
                st.error(f"Something went wrong: {e}")