from tempfile import NamedTemporaryFile
import os

import streamlit as st
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.readers.file import PDFReader
from dotenv import load_dotenv

load_dotenv()

# Set Streamlit page configuration
st.set_page_config(
    page_title="Resume & Cover Letter Feedback",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

# Initialize the chat messages history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome! Upload your resume or cover letter for feedback."}
    ]

# File upload section
st.sidebar.header("Upload Document")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["pdf"])

# Initialize OpenAI model
llm = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
    model="gpt-3.5-turbo",
    temperature=0.0,
    system_prompt="You are an expert on evaluating and giving feedback to resumes and cover letters. Provide detailed answers to these documents. Use the document to support your answers.",
)

# Initialize VectorStoreIndex
index = None

# Process uploaded file and create chat engine
if uploaded_file:
    bytes_data = uploaded_file.read()
    with NamedTemporaryFile(delete=False) as tmp:
        tmp.write(bytes_data)
        with st.spinner("Loading and indexing the document..."):
            reader = PDFReader()
            docs = reader.load_data(tmp.name)
            index = VectorStoreIndex.from_documents(docs)
    os.remove(tmp.name)

    # Initialize chat engine
    if index:
        st.session_state.chat_engine = index.as_chat_engine(
            chat_mode="condense_question", verbose=False, llm=llm
        )

# Chat interface section
st.title("Resume & Cover Letter Feedback")
if uploaded_file:
    st.subheader("Document Upload Complete")
    st.markdown("You can now ask questions or provide prompts for feedback.")
    st.sidebar.write("✅ Document uploaded successfully.")
else:
    st.sidebar.write("❗ Please upload a document.")

# User input section
user_input = st.text_input("Ask me anything or provide feedback:", "")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate response from chat engine
    if index:
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.stream_chat(user_input)
            st.write_stream(response.response_gen)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)

# Display chat messages
for message in st.session_state.messages:
    with st.container():
        st.markdown(f"**{message['role'].capitalize()}:** {message['content']}")
