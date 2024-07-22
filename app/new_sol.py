import streamlit as st
from llama_index.llms.openai import OpenAI
import pdfplumber
from llama_index.core import (
    VectorStoreIndex,
    ServiceContext,
    StorageContext
)
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import Document
import requests

# streamlit sidebar
st.sidebar.header("Slack Config")
slack_channel_key =st.sidebar.text_input("Slack Channel Key") 
slack_api_key = st.sidebar.text_input("Slack API Key") 

# send response to slack
def send_to_slack(chat):

    if  slack_api_key and slack_channel_key:

            slack_url = 'https://slack.com/api/chat.postMessage'

            headers = {'Authorization': f'Bearer {slack_api_key}',
                    'Content-Type': 'application/json'
                    }
            payload = {'channel': slack_channel_key,
                    'text': chat
                    }
            response = requests.post(slack_url, headers=headers, json=payload).json()

            if response["ok"]:
                st.success("Sent!")
               
           
    else:
        st.write("Add Slack Configuration keys!")
# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_text = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            pdf_text.append(Document(text=page.extract_text()))
    return pdf_text

def handle_questions(pdf_text, questions_list):
    # Parse the document into nodes of specific chunk size
    node_parser = SimpleNodeParser.from_defaults(chunk_size=500, chunk_overlap=20)
    #nodes = node_parser.parse(pdf_text)

    # Init OpenAI models
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
    embed_model = OpenAIEmbedding(model="text-embedding-ada-002", embed_batch_size=100)

    # Set vector db context
    service_context = ServiceContext.from_defaults(
        llm=llm, embed_model=embed_model, node_parser=node_parser
    )

    # Use a simple in-memory vector store
    vector_store = SimpleVectorStore()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Convert document to OpenAI embeddings and upsert into vector store
    index = VectorStoreIndex.from_documents(
        documents=pdf_text,
        storage_context=storage_context,
        service_context=service_context,
        show_progress=True,
    )

    # Create query engine and get responses
    query_engine = index.as_query_engine()
    answers = {}
    for question in questions_list:
        response = query_engine.query(question)
        answers[question] = response.response if response.response else "Data Not Available"
    
    return answers

# Streamlit interface
st.title("PDF Question Answering with LlamaIndex and OpenAI")

# Upload PDF
pdf_file = st.file_uploader("Upload your PDF", type="pdf")
questions = st.text_area("Enter your questions (one per line)")

# Process input
if pdf_file and questions:
    questions_list = questions.strip().split('\n')
    if not questions_list:
        st.error("Please enter at least one question.")
    else:
        # Extract text from PDF
        pdf_text = extract_text_from_pdf(pdf_file)
        
        # Process questions and generate answers
        answers = handle_questions(pdf_text, questions_list)
        
        # Display results
        st.json(answers)
        if st.button("Send to Slack"):
            send_to_slack(answers)
