import streamlit as st
from agno.agent import Agent
from agno.embedder.google import GeminiEmbedder
from agno.knowledge.website import WebsiteKnowledgeBase
from agno.models.google import Gemini
from agno.vectordb.pgvector import PgVector
from textwrap import dedent
import time

# Database URL and other configurations
db_url = "postgresql+psycopg2://postgres:c2FNJ5zNMGNZb2svEFc4@database-1.czg44aga0cfb.ap-south-1.rds.amazonaws.com:5432/ai"
# db_url = "postgresql+psycopg://ai:ai@localhost:5432/ai"

# Initialize KnowledgeBase and Agent
knowledge_base = WebsiteKnowledgeBase(
    urls=["https://www.adanifoundation.org", "https://www.adanifoundation.org/about-us", "https://www.adanifoundation.org/Our-Work", "https://www.adanifoundation.org/Our-Work/Education", "https://www.adanifoundation.org/Our-Work/Community-Infrastructure", "https://www.adanifoundation.org/Our-Work/Climate-Action", "https://www.adanifoundation.org/Our-Work/Health", "https://www.adanifoundation.org/Our-Work/Sustainable-Livelihood", "https://www.adanifoundation.org/Newsroom", "https://www.adanifoundation.org/Contact-Us"],
    max_links=7,
    vector_db=PgVector(
        table_name="adani_kb",
        db_url=db_url,
        embedder=GeminiEmbedder(api_key='AIzaSyBZPISHXJRqTuha1TXtoSV2fgZtshvtxFA'),
    ),
)
# knowledge_base.load(recreate=True)  # Comment out after first run

# Agent description and instructions
agent = Agent(
    model=Gemini(id="gemini-2.0-flash"),
    description="""\
    You are representing Adani Foundation, an AI Agent.
    Your goal is to provide information from the vector DB.
    """,
    instructions=dedent("""\
    1. Analyze the request.
    2. Search your knowledge base for relevant information.
    3. Present the information to the user.
    4. Provide concise, detailed but accurate answers based on the context.
    5. Do not make up or infer information that is not in the context.
    6. If the information needed is not available in the provided context, respond with "I don't have enough information to answer this question accurately."
    """),
    knowledge=knowledge_base,
)

# Streamlit UI
st.set_page_config(page_title="Adani Foundation Virtual Assistant", page_icon="🤖", layout="wide")
st.title("Adani Foundation Virtual Assistant")

# Define session state variables
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
    
if 'question' not in st.session_state:
    st.session_state['question'] = ""

# Function to handle user input
def handle_input():
    question = st.session_state.input_field
    if question:
        # Add user message to chat history
        st.session_state['messages'].append({"role": "user", "content": question})
        
        # Get response from agent
        with st.spinner('Thinking...'):
            time.sleep(1)  # Simulate typing delay
            response_object = agent.run(question, markdown=True)
            response_text = response_object.content
        
        # Add agent response to chat history
        st.session_state['messages'].append({"role": "assistant", "content": response_text})
        
        # Clear the input field
        st.session_state.input_field = ""

# Display chat messages
for message in st.session_state['messages']:
    if message['role'] == 'user':
        st.chat_message("user").markdown(message['content'])
    else:
        st.chat_message("assistant").markdown(message['content'])

# Input field with callback
st.text_input(
    "Ask a question:", 
    key="input_field",
    placeholder="Type your question here...",
    on_change=handle_input
)
