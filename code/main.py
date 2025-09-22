## integrate our code OpenAi API
import os
from constants import openai_key
from langchain.llms import OpenAI


import streamlit as st

os.environ["api_key"]=openai_key


# streamlit framework

st.title("LangChain Demo With OpenAI API")
input_text=st.text_input("Search The Topic You Want")


## OpenAI LLMS

llm = OpenAI(temperature=0.8, openai_api_key=openai_key)


if input_text:
    st.write(llm(input_text))

