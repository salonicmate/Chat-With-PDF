import os
# from langchain.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_pinecone import Pinecone as PC
import argparse
import time
import asyncio
import streamlit as st
import random
from dotenv import load_dotenv
load_dotenv()

# Index name in Pinecone database
index_name = "india-gdp-fy25"

def read_doc(directory):
    file_loader=PyPDFDirectoryLoader(directory)
    documents=file_loader.load()
    return documents

def chunk_data(docs,chunk_size=250,chunk_overlap=10):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    doc_splitted=text_splitter.split_documents(docs)
    return doc_splitted

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def chat_with_llm(query):
    doc=read_doc('documents/')
    chunks=chunk_data(docs=doc)
    vector_database_index = PC.from_documents(index_name = index_name, 
                                            documents = chunks, 
                                            embedding = embeddings)
    retriever = vector_database_index.as_retriever()

    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.6
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages= True)
    chain = ConversationalRetrievalChain.from_llm(gemini_llm, retriever= retriever, memory= memory)

    # query = "What was the crisil joshi's statement about the GDP"
    # answer = chain.run({'question': query})
    answer = chain.invoke({'question': query})
    print(answer['answer'])
    response = answer['answer']
    return response

def response_generator():
    response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

st.set_page_config(page_title='PDF Summarizer')
st.header('Ask to your Document')
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask to your custom document"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if prompt.lower() in ["hi",'hello']:
            response_llm = st.write_stream(response_generator())
        else:
            response_llm = chat_with_llm(prompt)
            async def reference_generator():
                for word in response_llm.split(" "):
                    yield f"{word} "
                    await asyncio.sleep(0.05)
            st.write_stream(reference_generator())
    st.session_state.messages.append({"role": "assistant", "content": response_llm})