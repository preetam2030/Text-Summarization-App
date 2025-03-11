#!/usr/bin/env python
# coding: utf-8

# In[14]:


import os
import getpass
import streamlit as st
from langchain_openai import OpenAI
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain

import tiktoken # to count the tokens


# In[16]:


# Set your Google API key
# def _set_env(var: str):
#     if not os.environ.get(var):
#         import getpass
#         os.environ[var] = getpass.getpass(f"{var}: ")

# _set_env("GOOGLE_API_KEY")
# api_key = os.environ.get("GOOGLE_API_KEY")
api_key = st.secrets["GOOGLE_API_KEY"]
if not api_key:
    st.error("Please provide a Google API key to use this application.")
    st.stop()
genai.configure(api_key=api_key)


# In[43]:


#encoder = tiktoken.get_encoding("cl100k_base")


# In[25]:


if 'tokens_consumed' not in st.session_state:
    st.session_state.tokens_consumed = 0  # Initialize total token count if not present
if 'query_tokens' not in st.session_state:
    st.session_state.query_tokens = 0     # Initialize query token count if not present
if 'response_tokens' not in st.session_state:
    st.session_state.response_tokens = 0  # Initialize response token count if not present


# In[81]:


llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
        max_tokens=800
    )

# Define the prompt template
prompt_template = """
You are a helpful assistant. Given the following context, summarize the text:

{context}
"""

# Create a PromptTemplate instance
prompt = PromptTemplate(input_variables=["context"], template=prompt_template)

# Initialize the LLMChain with the prompt and LLM
llm_chain = LLMChain(prompt=prompt, llm=llm)


# In[7]:


def generate_response(txt):
    # Instantiate the LLM model
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
        max_tokens=800
    )
    # Split text
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(txt)
    
    # Create multiple documents
    docs = [Document(page_content=t) for t in texts]
    
    # Text summarization
    chain = load_summarize_chain(llm, chain_type='map_reduce')
    return chain.run(docs)


# In[90]:


st.set_page_config(page_title="Text Summarization App")

# Display the title
st.title("Text Summarization App")

# Provide instructions
st.write("Upload a text file or enter text below to generate a summary:")

# File uploader
uploaded_file = st.file_uploader("Choose a text file", type=["txt"])

# Text input
text_input = st.text_area("Or enter text directly:")

# Process the text
if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")
elif text_input:
    text = text_input
else:
    text = None


query_tokens = 0
response_tokens = 0

if text:
    # Display the original text
    st.subheader("Original Text:")
    st.write(text)

    # Generate and display the summary
    st.subheader("Summary:")
    with st.spinner('Generating summary...'):
        summary = generate_response(text)
    st.write(summary)
   # Simplified token counting for Gemini (approximated based on characters)
    query_tokens = len(text) // 4
    response_tokens = len(summary) // 4


# In[93]:


st.session_state.query_tokens += query_tokens
st.session_state.response_tokens += response_tokens
st.session_state.tokens_consumed += (query_tokens + response_tokens)
st.sidebar.write(f"Total Tokens Consumed: {st.session_state.tokens_consumed}")
st.sidebar.write(f"Query Tokens: {st.session_state.query_tokens}")
st.sidebar.write(f"Response Tokens: {st.session_state.response_tokens}")

print("Tokens consumed in this transaction...")
print("Query token = " , query_tokens)
print("Response tokens = " , response_tokens)
st.session_state.tokens_consumed = 0
st.session_state.query_tokens = 0
st.session_state.response_tokens= 0


# In[ ]:




