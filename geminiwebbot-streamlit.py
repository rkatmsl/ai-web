import streamlit as st
from agno.agent import Agent
from agno.embedder.google import GeminiEmbedder
from agno.knowledge.website import WebsiteKnowledgeBase
from agno.models.google import Gemini
from agno.vectordb.pgvector import PgVector
from textwrap import dedent
import time
import os

pg_pass = st.secrets["PG_PASS"]
db_url = f"postgresql://neondb_owner:{pg_pass}@ep-dry-boat-a56osczd-pooler.us-east-2.aws.neon.tech/neondb?sslmode=require"

# Streamlit UI setup - Place this at the beginning
st.set_page_config(page_title="Adani Foundation Virtual Assistant", page_icon="🤖", layout="wide")
st.title("Adani Foundation Virtual Assistant")

# Initialize session state variables - Make sure these are defined before any UI elements
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if 'knowledge_base_initialized' not in st.session_state:
    # Initialize KnowledgeBase
    knowledge_base = WebsiteKnowledgeBase(
    urls=["https://www.adanifoundation.org", "https://www.adanifoundation.org/about-us", "https://www.adanifoundation.org/Our-Work", "https://www.adanifoundation.org/Our-Work/Education", "https://www.adanifoundation.org/Our-Work/Community-Infrastructure", "https://www.adanifoundation.org/Our-Work/Climate-Action", "https://www.adanifoundation.org/Our-Work/Health", "https://www.adanifoundation.org/Our-Work/Sustainable-Livelihood", "https://www.adanifoundation.org/Newsroom", "https://www.adanifoundation.org/Contact-Us"],
    max_links=7, # Reduced max_links for faster initial loading, adjust as needed
        vector_db=PgVector(
            table_name="adani_kb",
            db_url=db_url,
            embedder=GeminiEmbedder(),
        ),
    )
    # knowledge_base.load(recreate=True)
    st.session_state['knowledge_base'] = knowledge_base
    st.session_state['knowledge_base_initialized'] = True

# Function to build conversation context
def build_conversation_context(messages):
    if not messages:
        return ""

    context = "Previous conversation:\n"
    for msg in messages:
        prefix = "User: " if msg["role"] == "user" else "Assistant: "
        context += f"{prefix}{msg['content']}\n\n"

    return context

# Create agent with conversation context
def get_agent_with_context():
    # Get all messages except the most recent user message (which will be the input to the agent)
    if len(st.session_state['messages']) > 0 and st.session_state['messages'][-1]['role'] == 'user':
        history = st.session_state['messages'][:-1]
    else:
        history = st.session_state['messages']

    context = build_conversation_context(history)

    return Agent(
        model=Gemini(id="gemini-2.0-flash", api_key=API_KEY),
        description="""
        You are representing Adani Foundation, an AI Agent.
        Your goal is to provide information from the vector DB related to Adani Foundation.
        """,
        instructions=dedent(f"""
        1. Analyze the request from the user.
        2. Search your knowledge base for relevant information about Adani Foundation based on the request.
        3. Present the information to the user in a clear and concise manner.
        4. Provide detailed but accurate answers based on the context retrieved from the knowledge base.
        5. Do not make up or infer information that is not explicitly present in the retrieved context.
        6. If the information needed is not available in the provided context, respond with "I don't have enough information to answer this question accurately from my knowledge base."
        7. Maintain a conversational tone and refer to previous parts of the conversation when relevant to maintain context.
        8. Remember details that the user has shared previously in the conversation to provide more personalized and relevant responses.

        {context}
        """),
        knowledge=st.session_state['knowledge_base'],
    )

# Display chat messages
for message in st.session_state['messages']:
    if message['role'] == 'user':
        st.chat_message("user").markdown(message['content'])
    else:
        st.chat_message("assistant").markdown(message['content'])

# Function to handle user input
def handle_input():
    question = st.session_state.input_field
    if question:
        # Add user message to chat history
        st.session_state['messages'].append({"role": "user", "content": question})

        # Get response from agent with conversation context
        with st.spinner('Thinking...'):
            # Get a fresh agent instance with updated conversation history
            contextual_agent = get_agent_with_context()

            response_object = contextual_agent.run(question, markdown=True)
            response_text = response_object.content

        # Add agent response to chat history
        st.session_state['messages'].append({"role": "assistant", "content": response_text})

        # Clear the input field
        st.session_state.input_field = ""

# Input field with callback
st.text_input(
    "Ask a question about Adani Foundation:", # More specific placeholder
    key="input_field",
    placeholder="Type your question here...",
    on_change=handle_input
)

# Optional - Add a button to clear conversation history
if st.button("Clear Conversation"):
    st.session_state['messages'] = []
    st.rerun()
