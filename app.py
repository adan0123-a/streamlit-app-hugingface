import streamlit as st
from transformers import pipeline
from PIL import Image

st.set_page_config(page_title="🤖 AI Toolbox", layout="centered")
st.title("🤗 Hugging Face Streamlit App")

# Sidebar for model choice
task = st.sidebar.radio("Choose a Task", ["Text Generation", "Visual QA", "Text Summarization"])

# ----------------- TEXT GENERATION -----------------
if task == "Text Generation":
    st.subheader("📝 Text Generation (GPT-2)")
    prompt = st.text_area("Enter a prompt", "Once upon a time in a land far away,")
    if st.button("Generate Text"):
        with st.spinner("Generating..."):
            generator = pipeline("text-generation", model="openai-community/gpt2")
            output = generator(prompt, max_length=100, num_return_sequences=1)
            st.success("Generated Text:")
            st.write(output[0]['generated_text'])

# ----------------- VISUAL QUESTION ANSWERING -----------------
elif task == "Visual QA":
    st.subheader("🖼️ Visual Question Answering")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    question = st.text_input("Ask a question about the image", "What colors are used in this image?")
    
    if uploaded_image and question:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Get Answer"):
            with st.spinner("Answering..."):
                vqa = pipeline("visual-question-answering", model="dandelin/vilt-b32-finetuned-vqa")
                result = vqa(image, question)
                st.success(f"Answer: {result[0]['answer']}")

# ----------------- TEXT SUMMARIZATION -----------------
elif task == "Text Summarization":
    st.subheader("📚 Text Summarization")
    input_text = st.text_area("Paste long text here", height=200)
    if st.button("Summarize"):
        with st.spinner("Summarizing..."):
            summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
            summary = summarizer(input_text, max_length=130, min_length=30, do_sample=False)
            st.success("Summary:")
            st.write(summary[0]['summary_text'])

