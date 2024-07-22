import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import (
    ServiceContext,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone
from llama_index.readers.file import PDFReader

# load all env variables
load_dotenv()

# init pinecone vector db
pc = Pinecone(api_key=os.environ["PINECONE_API"])

index_name = "llama-docs"

pinecone_index = pc.Index(index_name)

if __name__ == "__main__":
    # fetch all documents from directory and load the pdf files  
    dir_reader = SimpleDirectoryReader(
        input_dir="../data", file_extractor={".pdf": PDFReader()}
    )

    documents = dir_reader.load_data()

    # parse documents into nodes of of specific chunk side ***IMPORTANT
    node_parser = SimpleNodeParser.from_defaults(chunk_size=500, chunk_overlap=20)

    # init model
    llm = OpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
    )

    # init embedding model
    embed_model = OpenAIEmbedding(model="text-embedding-ada-002", embed_batch_size=100)

    # set for vector db context 
    service_context = ServiceContext.from_defaults(
        llm=llm, embed_model=embed_model, node_parser=node_parser
    )

    # set pinecone vector db
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    # set storage context 
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # convert document to openai embedding (encode-> context embedding-> positional embedding-> nodes metadata)  
    # upsert embeddings in pinecone 
    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        service_context=service_context,
        show_progress=True,
    )


