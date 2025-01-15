### Chat with Custom PDF

This LLM application is based on RAG - Retrieval-Augmented Generation
Pinecone vector database is used to store the vector index
Vector index is created in the Pinecone space

### Steps to follow

1. Run the file create_vector_index.py to create vector index in the Pinecone space
   
	python run create_vector_index.py -create_index -index_name your-index-name
3. To chat with PDF start the streamlit application
   
	streamlit run ChatWithPDF.py
	

