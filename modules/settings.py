from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma

import chromadb

ollamaBaseURL = "http://localhost:11434"
ollamaModel = "mistral"

def init():
    global initialized

    global oembed
    global chroma

    try:
        initialized
    except NameError:
        oembed = OllamaEmbeddings(base_url = ollamaBaseURL, model = ollamaModel)

        chroma = Chroma(persist_directory = "ChromaDB", embedding_function = oembed, collection_name = "ollama-chatpdf")
        initialized = True
