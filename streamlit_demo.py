import streamlit as st
import pandas as pd
import numpy as np
import random
import time
from rag_backend import rag_stream


st.title('RAG_demo')


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if query := st.chat_input("Ask a question about Professor Whinston's papers"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(query)

# Display assistant response in chat message container
    with st.chat_message("assistant"):
        
        
        
        answer, doc = rag_stream(query)
        st.markdown(answer)
        st.divider()
        
        for i, d in enumerate(doc, 1):
            st.markdown(f"source{i}. {d.metadata.get('source')}, page{d.metadata.get('page')}")
            st.markdown(d.page_content[:500] + " …")
            st.divider()
        
        


    st.session_state.messages.append({"role": "assistant", "content": answer})                