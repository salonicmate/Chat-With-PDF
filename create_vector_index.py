import os
# from langchain.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_pinecone import PineconeVectorStore
from langchain_pinecone import Pinecone as PC
import argparse
from dotenv import load_dotenv
load_dotenv()



parser = argparse.ArgumentParser()
parser.add_argument('-create_index', action='store_true')
parser.add_argument('-index_name', type=str, default='india-gdp-fy25')
args = parser.parse_args()

index_name = args.index_name
if args.create_index:
    from pinecone import ServerlessSpec, Pinecone
    import time
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    if not pc.has_index(index_name):  
        print("Index does not exsist, creating new")  
        pc.create_index(
            name=index_name,
            # dimension=1024,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws', 
                region='us-east-1'
            ) 
        ) 

    # Wait for the index to be ready
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)