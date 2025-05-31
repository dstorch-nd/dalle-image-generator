import streamlit as st
from openai import OpenAI
import base64
from io import BytesIO
from PIL import Image

# Initialize OpenAI client using secret key in Streamlit
OpenAI.api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI()

st.set_page_config(page_title="GPT Image Chat", layout="wide")
st.title("🧠 GPT-Image Chat — Generate & Edit Images")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of dicts with keys: role, type, content, revised_prompt, image_b64, response_id
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
        "model": "gpt-image-1",
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

# UI form for input
with st.form("input_form", clear_on_submit=True):
    user_text = st.text_area("Enter your message or prompt here:", height=100)
    submitted = st.form_submit_button("Send")

if submitted:
    if not user_text.strip():
        st.warning("Please enter some text.")
    else:
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "type": "text",
            "content": user_text,
        })

        # Generate or edit image based on last response ID
        with st.spinner("Generating image or editing..."):
            image_b64, revised_prompt, response_id = generate_image(user_text, st.session_state.last_response_id)

        if image_b64:
            # Add assistant image message
            st.session_state.messages.append({
                "role": "assistant",
                "type": "image",
                "content": "",
                "revised_prompt": revised_prompt,
                "image_b64": image_b64,
                "response_id": response_id,
            })
            st.session_state.last_response_id = response_id
        else:
            # If no image, fallback text
            st.session_state.messages.append({
                "role": "assistant",
                "type": "text",
                "content": "Sorry, I couldn't generate an image for that.",
            })

# Display chat history
st.markdown("---")
st.markdown("### Chat History")
for msg in st.session_state.messages:
    render_message(msg)
    st.markdown("---")
