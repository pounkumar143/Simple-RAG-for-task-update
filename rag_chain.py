import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# Set your Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Load PDF documents
def load_docs(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

# Create vector store from documents
def create_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# Initialize the LLM
def get_llm():
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama3-8b-8192"
    )

# Create the RAG chain
def get_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = get_llm()
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return rag_chain

# Streamlit UI
st.title("📄 RAG Chatbot with PDF Upload")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF uploaded successfully!")

    docs = load_docs("uploaded.pdf")
    vectorstore = create_vectorstore(docs)
    rag_chain = get_rag_chain(vectorstore)

    user_input = st.text_input("Ask a question about the document:")

    if st.button("Submit"):
        if user_input:
            response = rag_chain.run(user_input)
            st.markdown(f"**Response:** {response}")
        else:
            st.warning("Please enter a question.")
