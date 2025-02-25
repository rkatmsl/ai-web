import streamlit as st
from agno.agent import Agent
from agno.embedder.google import GeminiEmbedder
from agno.knowledge.website import WebsiteKnowledgeBase
from agno.models.google import Gemini
from agno.vectordb.pgvector import PgVector
from textwrap import dedent

# Database URL and other configurations
db_url = "postgresql+psycopg://ai:ai@localhost:5432/ai"

# Initialize KnowledgeBase and Agent
knowledge_base = WebsiteKnowledgeBase(
    urls=["https://www.adanifoundation.org", "https://www.adanifoundation.org/about-us", "https://www.adanifoundation.org/Our-Work", "https://www.adanifoundation.org/Our-Work/Education", "https://www.adanifoundation.org/Our-Work/Community-Infrastructure", "https://www.adanifoundation.org/Our-Work/Climate-Action", "https://www.adanifoundation.org/Our-Work/Health", "https://www.adanifoundation.org/Our-Work/Sustainable-Livelihood", "https://www.adanifoundation.org/Newsroom", "https://www.adanifoundation.org/Contact-Us"],
    max_links=7,
    vector_db=PgVector(
        table_name="website_documents",
        db_url=db_url,
        embedder=GeminiEmbedder(),
    ),
)

# Agent description and instructions
agent = Agent(
    model=Gemini(id="gemini-1.5-pro"),
    description="""\
    You are representing Adani Foundation, an advanced AI Agent.
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
    # show_tool_calls=True,
)

# Streamlit UI
st.title("Adani Foundation Virtual Assistant")

# Input field for the user to type a question
question = st.text_input("Ask a question:", "")

# Button to get the response
if st.button("Get Answer"):
    if question:
        # Pass the question to the agent and get the response
        response_object = agent.run(question, markdown=True) # Get the RunResponse object
        response_text = response_object.content # Extract the content string
        # Show the agent's response
        st.write("Agent's Response:")
        st.markdown(response_text) # Use response_text here
    else:
        st.warning("Please ask a question.")
