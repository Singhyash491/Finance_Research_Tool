import os
os.environ["PYTHONPATH"] = os.getcwd()
os.environ["PYTORCH_JIT"] = "0"

import streamlit as st
import time
import logging
import traceback
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

# Load environment variables
load_dotenv()

# Configure Google Gemini API
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {str(e)}")

# Set up the page
st.title("💬 Finance Research Tool")

# Sidebar configuration
with st.sidebar:
    st.title("Configuration")
    urls = [st.text_input(f"URL {i+1}") for i in range(3)]
    process_url_clicked = st.button("Process URLs")
    st.divider()
    st.markdown("## How to Use")
    st.markdown("1. Enter URLs in the sidebar\n2. Click 'Process URLs'\n3. Start chatting below!")
    st.divider()
    if st.checkbox("Show Debug Info"):
        st.write("## Debug Information")
        st.json({
            "conversation_length": len(st.session_state.conversation),
            "vectorstore_initialized": st.session_state.vectorstore is not None
        })

# Display chat messages
for message in st.session_state.conversation:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 Sources"):
                for source in message["sources"]:
                    st.caption(f"🔗 {source['metadata']['source']}")
                    st.write(source["page_content"][:300] + "...")

# Fixed chat input at bottom
if prompt := st.chat_input("Ask your question about the documents..."):
    # Add user message to chat history
    st.session_state.conversation.append({"role": "user", "content": prompt})
    
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Process response
    if st.session_state.vectorstore:
        try:
            with st.spinner("Analysing documents..."):
                # Create QA chain
                qa_chain = RetrievalQA.from_chain_type(
                    llm=ChatGoogleGenerativeAI(
                        model="gemini-1.5-flash-latest",
                        temperature=0.7,
                        max_output_tokens=500
                    ),
                    chain_type="stuff",
                    retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3}),
                    return_source_documents=True
                )
                
                # Get response
                response = qa_chain({"query": prompt})
                
                # Format sources
                sources = [{
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in response["source_documents"]]
                
                # Add assistant response to history
                st.session_state.conversation.append({
                    "role": "assistant",
                    "content": response["result"],
                    "sources": sources
                })
                
                # Display response
                with st.chat_message("assistant"):
                    st.markdown(response["result"])
                    with st.expander("📚 Sources"):
                        for source in sources:
                            st.caption(f"🔗 {source['metadata']['source']}")
                            st.write(source["page_content"][:300] + "...")
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            st.session_state.conversation.append({
                "role": "assistant",
                "content": "❌ Error processing request. Please check the URLs and try again."
            })
            logger.error(traceback.format_exc())
    else:
        st.session_state.conversation.append({
            "role": "assistant",
            "content": "⚠️ Please process URLs first using the sidebar!"
        })

# URL processing
if process_url_clicked:
    valid_urls = [url for url in urls if url]
    
    if not valid_urls:
        st.sidebar.error("Please enter at least one valid URL")
    else:
        try:
            with st.sidebar.status("Processing URLs...", expanded=True) as status:
                st.write("📥 Downloading documents...")
                loader = UnstructuredURLLoader(urls=valid_urls)
                data = loader.load()
                
                st.write("📖 Splitting text...")
                text_splitter = RecursiveCharacterTextSplitter(
                    separators=['\n\n', '\n', '.', ','],
                    chunk_size=1000
                )
                docs = text_splitter.split_documents(data)
                
                st.write("🧠 Creating embeddings...")
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                st.session_state.vectorstore = FAISS.from_documents(docs, embeddings)
                
                status.update(label="✅ Processing complete!", state="complete", expanded=False)
            
            # Add system message
            st.session_state.conversation.append({
                "role": "assistant",
                "content": "I'm ready to answer questions about the provided documents!",
                "sources": []
            })
            
        except Exception as e:
            st.sidebar.error(f"Error processing URLs: {str(e)}")
            logger.error(traceback.format_exc())