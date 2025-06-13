# retriever_setup.py

import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.utils.utils import convert_to_secret_str

# --- RAG Setup: Load Vector Store and Create Retriever ---
# This function encapsulates all RAG setup (embeddings, Chroma load, retriever config)
@st.cache_resource(show_spinner="Loading AI knowledge base...") # Show spinner while loading
def get_retriever(persist_directory: str, collection_name: str, _openai_api_key):
    print("--- DEBUG retriever_setup: Inside get_retriever function ---") 
    
    # Use the SecretStr object directly
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large", api_key=_openai_api_key)
    
    try:
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings_model,
            collection_name=collection_name
        )
        actual_doc_count = vector_store._collection.count()
        print(f"--- DEBUG retriever_setup: Vector store loaded. Actual doc count: {actual_doc_count} ---") 
    except Exception as e:
        st.error(f"Could not load Chroma vector store. Ensure '{persist_directory}' directory exists and is accessible. Error: {e}")
        st.stop()
        
    RETRIEVAL_K = min(3, actual_doc_count) # Set k to 3, but not more than available
    print(f"--- DEBUG retriever_setup: Retrieval K set to {RETRIEVAL_K} ---") 

    retriever = vector_store.as_retriever(
        search_type="mmr", 
        search_kwargs={
            "k": RETRIEVAL_K,           
            "lambda_mult": 0.5,         
            "fetch_k": actual_doc_count # Fetch up to actual count to avoid warnings
        }
    )
    st.success(f"Alex's characteristics vector store loaded successfully! (Documents in store: {actual_doc_count}, Retrieving K={RETRIEVAL_K}, Fetch K={actual_doc_count})")
    print("--- DEBUG retriever_setup: Retriever configured ---") 
    return retriever