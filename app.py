import os
import streamlit as st
from rag_chain import load_docs, create_vectorstore, get_rag_chain

# Set your Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("📄 RAG Chatbot with PDF Upload")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("PDF uploaded successfully!")

    # Load and process the PDF
    docs = load_docs("uploaded.pdf")
    vectorstore = create_vectorstore(docs)
    rag_chain = get_rag_chain(vectorstore)

    # User input
    user_input = st.text_input("Ask a question about the document:")

    if st.button("Submit"):
        if user_input:
            response = rag_chain.run(user_input)
            st.markdown(f"**Response:** {response}")
        else:
            st.warning("Please enter a question.")
