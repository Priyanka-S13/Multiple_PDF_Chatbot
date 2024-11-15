# GenerativeAI

# Multipdf chatbot using gemini

In today's digital landscape, efficiently accessing and extracting 
pertinent information from PDF documents remains a significant 
challenge. Users across various domains struggle with manual scanning 
of documents and limited keyword-based search capabilities, leading to 
inefficiencies in information retrieval. This hampers productivity and 
slows down research, academic pursuits, and professional tasks. 
Therefore, there is a critical need for an intelligent PDF-based question 
answering system that automates information extraction, enhances user 
experience with intuitive querying interfaces, and revolutionizes 
knowledge management. By streamlining access to PDF information 
and improving search efficiency, this system addresses the pressing 
demand for advanced solutions in digital document interaction, offering 
substantial benefits in time savings, productivity, and user satisfaction.

## Features
- **Multipdf Support**: The chatbot can handle conversations that span across multiple PDF documents. It intelligently processes information from various sources to provide relevant responses.
- **Gemini Integration**: Utilizes the Gemini framework, a state-of-the-art conversational AI tool, for natural language processing and response generation. Gemini helps in understanding user queries effectively and generating contextually appropriate responses.
- **Dual Encoder Architecture**: The chatbot architecture includes a Dual Encoder system. This architecture consists of two separate encoders:
  - **Context Encoder**: Responsible for encoding the context information, which includes user queries, conversation history, and information extracted from PDF documents.
  - **Response Encoder**: Encodes the response candidates to facilitate accurate response generation.
  
# Download and preprocess PDF documents:

python app.py --pdf_dir /path/to/pdf_directory


## Installation

Install my-project with npm

```bash
streamlit
google-generativeai
python-dotenv
langchain
PyPDF2
chromadb
faiss-cpu
langchain_google_genai
```
    
## Tech Stack

**front-end:** streamlit python library

**back-end:** gemini api

Enter your query or conversation to interact with the chatbot. The chatbot will process the input and provide responses based on the context extracted from the multipdf documents.
Dual Encoder Architecture
The Dual Encoder Architecture is a key component of the chatbot system. It enhances the understanding of context and facilitates accurate response generation. Here's how it works:

* Context Encoder: 
This encoder is responsible for encoding the context information. It takes into account various factors such as user queries, conversation history, and information extracted from multipdf documents.
* Response Encoder:
 The Response Encoder encodes the response candidates. It prepares potential responses for comparison and selection.
These encoders are trained jointly using a dual learning objective. The model learns to maximize the similarity between the encoded context and the ground truth response while minimizing the similarity with incorrect responses. This training process ensures that the chatbot generates contextually appropriate and relevant responses.

## Contributing
Contributions are welcome! If you have any ideas, suggestions, or bug fixes, please feel free to submit issues and pull requests. Your contributions will help improve the functionality and effectiveness of the multipdf chatbot.


