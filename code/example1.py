## integrate our code OpenAi API
import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

from langchain.memory import ConversationBufferMemory

import streamlit as st

os.environ["api_key"]=openai_key


# streamlit framework

st.title("Celebrity Search Result ")
input_text=st.text_input("Search The Topic You Want")


# prompt templates


first_input_prompt=PromptTemplate(
    input_variables=["name"],
    template="Tell Me About Celebrity {name}"

)

# Memory

person_memory =ConversationBufferMemory(input_key="name",memory_key="chat_history")
dob_memory =ConversationBufferMemory(input_key="person",memory_key="chat_history")
descr_memory =ConversationBufferMemory(input_key="",memory_key="siscription_history")


## OpenAI LLMS

llm = OpenAI(temperature=0.8, openai_api_key=openai_key)
chain = LLMChain(llm=llm, prompt=first_input_prompt, verbose=True, output_key="person",memory=person_memory)



# prompt templates


second_input_prompt=PromptTemplate(
    input_variables=["person"],
    template="when was {person} born"

)


chain2 = LLMChain(
    llm=llm, prompt=second_input_prompt, verbose=True, output_key="dob",memory=dob_memory)



# prompt templates


third_input_prompt=PromptTemplate(
    input_variables=["dob"],
    template="mention 5 major events happend around {dob} in the world"

)

chain3 = LLMChain(llm=llm, prompt=third_input_prompt, verbose=True, output_key="description",memory=descr_memory)

parent_chain=SequentialChain(
    chains=[chain,chain2,chain3],input_variables=["name"],output_variables=["person","dob","description"],verbose=True)


if input_text:
    st.write(parent_chain({"name":input_text}))


    with st.expander("person name"):
        st.info(person_memory.buffer)

    with st.expander("Major Events"):
        st.info(descr_memory.buffer)
