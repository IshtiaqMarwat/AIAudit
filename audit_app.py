# -*- coding: utf-8 -*-
"""Audit Assistant App"""

# pip install -U streamlit langchain langchain-openai langchain-community faiss-cpu sentence-transformers PyPDF2

import streamlit as st
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# Load API Key from Streamlit secrets
if "OPENAI_API_KEY" not in st.secrets:
    st.error("❌ OpenAI API key not found. Please add it to Streamlit secrets.")
    st.stop()

openai_api_key = st.secrets["OPENAI_API_KEY"]

# Load FAISS index from current directory
VECTOR_DB_PATH = "."  # because index.faiss and index.pkl are in the root

@st.cache_resource
def load_vector_db():
    faiss_file = os.path.join(VECTOR_DB_PATH, "index.faiss")
    pkl_file = os.path.join(VECTOR_DB_PATH, "index.pkl")

    if not os.path.exists(faiss_file) or not os.path.exists(pkl_file):
        st.error("❌ FAISS index files not found. Ensure 'index.faiss' and 'index.pkl' are in the repo root.")
        st.stop()

    embeddings = HuggingFaceEmbeddings()
    return FAISS.load_local(VECTOR_DB_PATH, embeddings)

# Load the vector DB
db = load_vector_db()

# Initialize the OpenAI model
llm = ChatOpenAI(
    model="gpt-3.5-turbo",  # or "gpt-4o"
    temperature=0,
    api_key=openai_api_key
)

# Create QA chain
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False
)

# Streamlit UI
st.title("🕵️ AI Audit Assistant")
st.markdown("Ask any question related to the audit documents already loaded into the assistant.")

user_question = st.text_input("🔍 Ask your audit question:")

if user_question:
    with st.spinner("🔎 Finding answers..."):
        try:
            result = qa_chain.invoke(user_question)
            st.text_area("📘 Answer", value=result["result"], height=200)
        except Exception as e:
            st.error(f"❌ Error: {e}")
