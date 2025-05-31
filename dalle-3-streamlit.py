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

st.sidebar.header("Options")
use_gpt_assist = st.sidebar.checkbox("Use GPT-4 to assist with prompt (if image is uploaded)")

# --- Step 1: Input section ---
text_input = st.text_area("Your idea (text)", placeholder="Describe the image you want to create.", height=100)
uploaded_image = st.file_uploader("Optional reference image (if you want GPT to analyze it)", type=["jpg", "jpeg", "png"])

def encode_image(image_data):
    return base64.b64encode(image_data).decode("utf-8")

if st.button("Generate Image"):
    if not text_input and not uploaded_image:
        st.warning("Please enter a description or upload an image.")
    else:
        with st.spinner("Generating prompt and image..."):
            try:
                # Compose initial messages
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a creative assistant that crafts prompts for DALL·E 3 image generation. "
                            "If an image is provided, consider its contents to refine the prompt."
                        )
                    }
                ]

                if use_gpt_assist:
                    # If user wants GPT assistance and image uploaded, refine prompt
                    if uploaded_image:
                        # Encode image and add it as an image message
                        image = Image.open(uploaded_image)
                        if image.mode == 'RGBA':
                            image = image.convert("RGB")
                        buffered = io.BytesIO()
                        image.save(buffered, format="JPEG")
                        image_data = buffered.getvalue()
                        base64_image = encode_image(image_data)

                        user_content = text_input if text_input else "Please analyze the uploaded image."
                        messages.append({"role": "user", "content": user_content})
                        messages.append({
                            "role": "user",
                            "content": [{
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                            }]
                        })
                    else:
                        messages.append({"role": "user", "content": text_input})

                    # Use GPT-4o-mini to refine the prompt
                    chat_response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=messages
                    )
                    refined_prompt = chat_response.choices[0].message.content.strip()
                else:
                    # No GPT prompt assistance, just use user text as prompt
                    refined_prompt = text_input

                st.success("Final Prompt for Image Generation:")
                st.text_area("Prompt", refined_prompt, height=100)

                # --- Step 3: Generate image using GPT-image tool (via chat) ---
                # To generate images, you send a special "tool" message in the chat to trigger image generation
                image_generation_messages = [
                    {
                        "role": "user",
                        "content": refined_prompt
                    },
                    {
                        "role": "tool",
                        "name": "image_generation",
                        "content": {
                            "prompt": refined_prompt,
                            "n": 1,
                            "size": "1024x1024"
                        }
                    }
                ]

                # Call chat completion with GPT-image-1 (the image generation tool is behind this model)
                image_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=image_generation_messages
                )

                # The response's message content should contain image info (check the tool response)
                # Typically, the image URL is inside message.choices[0].message.content or message.tools_response

                # Extract image URL (depends on exact API; below is an example assumption)
                message_content = image_response.choices[0].message
                # For newer APIs, image URLs often come in message.content or a tool response field

                # For demonstration, parse image URL from message content if JSON
                import json
                try:
                    parsed = json.loads(message_content.content)
                    image_url = parsed.get("data", [{}])[0].get("url")
                except Exception:
                    # fallback if content is plain text with URL
                    image_url = message_content.content.strip()

                if image_url:
                    st.image(image_url, caption="Generated Image by DALL·E 3", use_container_width=True)
                else:
                    st.error("Failed to retrieve image URL from the response.")

            except Exception as e:
                st.error(f"Something went wrong: {e}")
