# import streamlit as st
# from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# # Load models lazily and cache them
# # @st.cache_resource
# def get_model(task):
#     if task == "Summarization":
#         pipe = pipeline("summarization", model="t5-small")
#     elif task == "Rephrasing":
#         return pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws")
#     elif task == "Grammar Correction":
#         pipe = pipeline("text2text-generation", model="vennify/t5-base-grammar-correction")
#         return {"tokenizer": tokenizer, "model": model}

# # UI layout
# st.title("📝 AI Text Assistant")
# task = st.selectbox("Choose Task", ["Grammar Correction", "Rephrasing", "Summarization"])
# user_input = st.text_area("Enter your text here:", height=200)

# # Process button
# if st.button("Process"):
#     if user_input.strip() == "":
#         st.warning("⚠️ Please enter some text.")
#     else:
#         with st.spinner("🤖 Processing..."):
#             model_obj = get_model(task)

#             # Handle each task accordingly
#             if task == "Summarization":
#                 result = model_obj(user_input, max_length=130, min_length=30, do_sample=False)[0]['summary_text']

#             elif task == "Rephrasing":
#                 result = model_obj(f"paraphrase: {user_input}")[0]['generated_text']

#             elif task == "Grammar Correction":
#                 tokenizer = model_obj["tokenizer"]
#                 model = model_obj["model"]
#                 input_ids = tokenizer.encode(user_input, return_tensors="pt", truncation=True)
#                 outputs = model.generate(input_ids, max_length=128, num_beams=5, early_stopping=True)
#                 result = tokenizer.decode(outputs[0], skip_special_tokens=True)

#             st.success("✅ Done!")
#             st.text_area("Output", value=result, height=200)
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

