# llm_chain.py
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

def create_base_chain():
    """Assembles the core LCEL chain using Groq."""
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a knowledgeable space astronaut."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_query}")
    ])
    
    # Initialize the model using the environment variable automatically
    model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.9)
    output_parser = StrOutputParser()
    
    # Return the assembled base assembly line
    return template | model | output_parser