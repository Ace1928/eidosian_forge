import streamlit as st
from langchain.llms import Replicate
import os
def generate_response(input_text):
    llama2_13b_chat = 'meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d'
    llm = Replicate(model=llama2_13b_chat, model_kwargs={'temperature': 0.01, 'top_p': 1, 'max_new_tokens': 500})
    st.info(llm(input_text))