import streamlit as st
from openai import OpenAI
import base64
from io import BytesIO
from PIL import Image
import uuid

# Initialize OpenAI client
OpenAI.api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI()
st.set_page_config(page_title="GPT Image Chat", layout="wide")
st.title("🧠 GPT-Image Chat — Generate & Edit Images")

# Session state for conversation
if "messages" not in st.session_state:
    # Each item: dict with keys: role ("user"/"assistant"), type ("text"/"image"), content (str), revised_prompt (str), image_b64 (str), response_id (str)
    st.session_state.messages = []
if "last_response_id" not in st.session_state:
    st.session_state.last_response_id = None

def render_message(msg):
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        if msg["type"] == "text":
            st.markdown(f"**Assistant:** {msg['content']}")
        elif msg["type"] == "image":
            st.markdown(f"**Assistant (Revised Prompt):** {msg.get('revised_prompt', '')}")
            image_bytes = base64.b64decode(msg["image_b64"])
            image = Image.open(BytesIO(image_bytes))
            st.image(image, use_column_width=True)

def generate_image(prompt, previous_response_id=None):
    tools = [{"type": "image_generation"}]

    params = {
        "model": "gpt-4o",
        "input": prompt,
        "tools": tools,
    }
    if previous_response_id:
        params["previous_response_id"] = previous_response_id

    response = client.responses.create(**params)

    # Extract image_generation_call output
    img_output = None
    for output in response.output:
        if output.type == "image_generation_call":
            img_output = output
            break
    if img_output is None:
        st.error("No image generated.")
        return None, None, None

    return img_output.result, img_output.revised_prompt, response.id

# UI for user input
with st.form("input_form", clear_on_submit=True):
    user_text = st.text_area("Enter your message or prompt here:", height=100)
    uploaded_file = st.file_uploader("Or upload an image to edit (optional):", type=["png", "jpg", "jpeg"])
    submitted = st.form_submit_button("Send")

if submitted:
    if not user_text and not uploaded_file:
        st.warning("Please enter some text or upload an image.")
    else:
        # Show user message
        user_message = {"role": "user", "type": "text", "content": user_text}
        st.session_state.messages.append(user_message)

        # Prepare prompt for API call
        prompt = user_text
        if uploaded_file:
            # encode image to base64
            img_bytes = uploaded_file.read()
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            # Append instruction to prompt to edit this image
            prompt += "\n\nEdit this image as described above."
        else:
            img_b64 = None

        # Call OpenAI to generate or edit image
        with st.spinner("Generating response..."):
            image_b64, revised_prompt, response_id = generate_image(prompt, st.session_state.last_response_id)

        if image_b64:
            assistant_message = {
                "role": "assistant",
                "type": "image",
                "content": "",
                "revised_prompt": revised_prompt,
                "image_b64": image_b64,
                "response_id": response_id,
            }
            st.session_state.messages.append(assistant_message)
            st.session_state.last_response_id = response_id
        else:
            # Fallback, if no image generated, just add text assistant message
            assistant_message = {
                "role": "assistant",
                "type": "text",
                "content": "Sorry, I couldn't generate an image for that.",
            }
            st.session_state.messages.append(assistant_message)

# Display chat messages
st.markdown("---")
st.markdown("### Chat History")
for msg in st.session_state.messages:
    render_message(msg)
    st.markdown("---")