from dotenv import load_dotenv
from pinecone import Pinecone
import os
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
import streamlit as st
import requests

# load all env variables
load_dotenv()

# set streamlit page configs
st.set_page_config(
    page_title="Chat with LlamaIndex",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

# set chat session 
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# streamlit page
st.title("Zania.io Assessment")
st.write("We aready have relevant pdf embedding stored in vector space on pinecone. Just ask your questions and send the responses to Slack ")

# streamlit sidebar
st.sidebar.header("Slack Config")
slack_channel_key = st.sidebar.text_input("Slack Channel Key") 
slack_api_key =st.sidebar.text_input("Slack API Key") 

# init pinecone vector db
pc = Pinecone(api_key=os.environ["PINECONE_API"])

index_name = "llama-docs"

pinecone_index = pc.Index(index_name)

vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

query_engine= index.as_query_engine()

# get query response
def handle_input(query):
    st.session_state["chat_history"].append({"text": query, "role":"user"})
    response = query_engine.query(query)
    text = response.response
    if text: 
        st.session_state["chat_history"].append({ "text": text, "role":"bot", "sent":False})
        st.rerun()

# send response to slack
def send_to_slack(i, chat):

    if  slack_api_key and slack_channel_key:

            slack_url = 'https://slack.com/api/chat.postMessage'

            headers = {'Authorization': f'Bearer {slack_api_key}',
                    'Content-Type': 'application/json'
                    }
            payload = {'channel': slack_channel_key,
                    'text': chat['text']
                    }
            response = requests.post(slack_url, headers=headers, json=payload).json()

            original_chat = chat

            if response["ok"]:
                st.session_state["chat_history"][i]={**chat,  "sent": True}
                st.rerun()
           
    else:
        st.write("Add Slack Configuration keys!")
    

# show chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
else:
    for i, chat in enumerate(st.session_state["chat_history"]):
        if chat["role"] == "user":
            st.write("**You:**")
            st.write(chat['text'])
        else:
            st.write("**Bot:**")
            st.write(chat['text'])

            if chat['sent'] == False:
                send_button =st.button("Send to Slack", key=i) 

                if send_button:
                    send_to_slack(i, chat)
            else:
                st.markdown(f"<span style='color:green;'> ✓ Sent to Slack</span>", unsafe_allow_html=True)
            
            st.write("")


# bottom query input
st.markdown("""
    <style>
    .fixed-bottom {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: white;
        padding: 10px;
        box-shadow: 0px -1px 10px rgba(0, 0, 0, 0.1);
    }
    .fixed-bottom input {
        width: 80%;
        padding: 10px;
    }
    .fixed-bottom button {
        width: 18%;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
st.markdown('<div class="fixed-bottom">', unsafe_allow_html=True)
st.write("Ask Your Query:")
query = st.text_input("Enter your query", key="input_text", label_visibility="collapsed")
submit_button = st.button("Submit")
st.markdown('</div>', unsafe_allow_html=True)

if submit_button and query:
    handle_input(query)










