from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langchain_community.llms.ollama import Ollama

import streamlit as st
import os
from dotenv import load_dotenv


load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
## Langsmith tracking
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are Donald Trump."),
        ("user", "Question: {question}")
    ]
)

## Streamlit framework
st.title("Langchain with Ollama: Llama3")
input_text = st.text_input("Search for the topic you want")

# Ollama: Llama2 LLM
llm = Ollama(model="llama3")
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))