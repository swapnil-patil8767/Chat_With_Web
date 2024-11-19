import os
import streamlit as st
import pickle
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import UnstructuredURLLoader
from dotenv import load_dotenv

# Import ChatGroq instead of OpenAI or Google Generative AI
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()  # Loads variables from .env file (e.g., GOOGLE_API_KEY, GROQ_API_KEY)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Configure Google Generative AI (for embedding generation)
import google.generativeai as genai
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the LLM using ChatGroq
llm = ChatGroq(
    temperature=0,
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-70b-versatile"
)

# Streamlit app setup
st.title("Chat With Website 🌐")
st.sidebar.title("Website URLs 🕸️")

# URL input fields
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"  # Path to save/load FAISS index

# Main processing placeholder
main_placeholder = st.empty()

def create_prompt_template():
    prompt_template = """
    You are analyzing a webpage 
    1. Extract and summarize all key information from the page, including any text, columns, and tables. 
    2. Organize the content into a structured format: main points, key sections, data from tables, and any important insights.
    3. Answer the following question based on the extracted contentss
    If the answer is not in the provided context, say, "Answer is not available in the context."
    
    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    return PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def build_faiss_index(docs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(docs, embedding=embeddings)
    vector_store.save_local("faiss_index")  # Save FAISS index locally
    return vector_store

def load_faiss_index():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

def get_qa_chain():
    prompt = create_prompt_template()
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain

if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    # Split data into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    # Create and save FAISS index
    main_placeholder.text("Building FAISS Embedding Vector...✅✅✅")
    vectorstore = build_faiss_index(docs)
    time.sleep(2)
    st.success("FAISS index created and saved successfully!")

# Query input and retrieval
query = main_placeholder.text_input("❀ Ask a Question: ")
if query:
    # Load the FAISS index and retrieve similar documents
    if os.path.exists("faiss_index"):
        vectorstore = load_faiss_index()
        docs = vectorstore.similarity_search(query)

        # Load the conversational chain and get the response
        chain = get_qa_chain()
        response = chain({"input_documents": docs, "question": query}, return_only_outputs=True)

        st.header("Answer ✨")
        st.write(response.get("output_text", "No response generated."))

        # Display sources if available
        sources = response.get("sources", "")
        if sources:
            st.subheader("Sources:")
            for source in sources.split("\n"):
                st.write(source)
