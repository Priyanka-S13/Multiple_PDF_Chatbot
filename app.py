import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not available, respond with
    "answer is not available in the context." Avoid providing incorrect answers.\n\n
    Context:\n{context}?\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("**Answer:**", response["output_text"])

def main():
    # Set page layout and theme
    st.set_page_config(page_title="GENAI PDF Chatbot 📄", page_icon="📚", layout="wide")

    # Custom CSS for styling
    st.markdown(
        """
        <style>
            /* Global Styling */
            body {background-color: #f0f4f8; font-family: 'Arial', sans-serif;}
            h1, h3 {font-family: 'Arial', sans-serif; color: #1e3c72;}
            .stTextInput input {
                background-color: #ffffff;
                color: #333333;  /* Ensure the text inside the input is dark and visible */
                border-radius: 8px;
                padding: 12px;
                font-size: 16px;
            }

            .stButton>button {
                background-color: #ff5722;
                color: white;
                border-radius: 8px;
                font-size: 16px;
                padding: 12px 30px;
                transition: background-color 0.3s ease;
            }
            .stButton>button:hover {
                background-color: #e64a19;
            }

            .stFileUploader>label {
                background-color: #e3f2fd;
                padding: 15px;
                border-radius: 8px;
                font-size: 16px;
                cursor: pointer;
                transition: background-color 0.3s ease;
            }

            .stFileUploader>label:hover {
                background-color: #bbdefb;
            }

            .title {
                text-align: center;
                font-size: 2.5em;
                color: #1e88e5;
                font-weight: bold;
                margin-top: -20px;
                background: linear-gradient(135deg, #67b7e8, #1e88e5);
                -webkit-background-clip: text;
                color: transparent;
            }

            .header-section {
                background-color: #2196f3;
                color: white;
                padding: 40px;
                border-radius: 8px;
                margin-bottom: 40px;
                text-align: center;
                box-shadow: 0px 8px 16px rgba(0, 0, 0, 0.1);
            }

            .footer {
                color: #555;
                font-size: 1em;
                text-align: center;
                margin-top: 30px;
                padding: 15px;
                background-color: #f7f8fc;
                border-radius: 10px;
                border-top: 2px solid #2196f3;
            }

            .emoji {
                font-size: 1.8em;
                margin-right: 10px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header Section with Emojis
    st.markdown("<div class='header-section'><h1><span class='emoji'>📚</span>GENAI PDF Chatbot <span class='emoji'>💬</span></h1><p>Upload PDFs, ask questions, and get accurate answers instantly!</p></div>", unsafe_allow_html=True)

    # Sidebar with Emojis
    with st.sidebar:
        st.header("📂 Document Processing 📑")
        pdf_docs = st.file_uploader("Upload PDF Files", type="pdf", accept_multiple_files=True)
        if st.button("Submit & Process 🚀", use_container_width=True):
            if pdf_docs:
                with st.spinner("Extracting and indexing text... 🔄"):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Text processed and indexed successfully! 🎉")
            else:
                st.warning("Please upload at least one PDF file to proceed. 🧐")

    # Main question interface with Emoji
    st.markdown("<div class='title'>Ask a Question 💡</div>", unsafe_allow_html=True)
    user_question = st.text_input("Type your question here 🤔", placeholder="E.g., What is the main topic of the document?")
    if user_question:
        with st.spinner("Searching for the answer... 🔍"):
            user_input(user_question)

    # Footer with Emoji
    st.markdown(
        "<div class='footer'>Instructions: Upload PDFs in the sidebar 📂, then type your question in the main area 💬 to receive contextually accurate responses. 🤖</div>",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
