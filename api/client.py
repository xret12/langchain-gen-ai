import requests
import streamlit as st


def get_openai_response(input_text):
    """Get an OpenAI response to the given input text."""
    response = requests.post("http://localhost:8000/essay/invoke",
                             json={
                                 "input":{"topic": input_text}
                             }
    )
    return response.json()["output"]["content"]



def get_ollama_response(input_text):
    """Get an Ollama response to the given input text."""
    response = requests.post("http://localhost:8000/poem/invoke",
                             json={
                                 "input":{"topic": input_text}
                             }
    )
    return response.json()["output"]


# Streamlit Framework
st.title("OpenAI and Ollama")
input_text1 = st.text_input("Write an essay on")
input_text2 = st.text_input("Write a poem on ")

if input_text1:
    st.write(get_openai_response(input_text1))

if input_text2:
    st.write(get_ollama_response(input_text2))


