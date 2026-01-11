import os
import logging
from typing import List

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)
logging.basicConfig(filename='app.log', format="%(asctime)s %(levelname)s %(message)s")

class DBManager:
    """
    Manages ChromaDB based knowledge base
    """

    def __init__(self, persist_directory: str = "./db"):
        self.db: Chroma | None = None
        self.collection_name: str = "doc_collection"

        # Create directory if it doesn't exist
        self.persist_directory: str = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)

        # Initialize embeddings
        self.embeddingModelName: str = "qwen3-embedding:4b"
        self.emb: OllamaEmbeddings = OllamaEmbeddings(model=self.embeddingModelName)

        self.create_or_load_db("user_collection")

    def create_or_load_db(self, collection_name: str):
        """ Creates new or loads existing database """
        try:
            self.db = Chroma(collection_name=collection_name,
                             embedding_function=self.emb,
                             persist_directory=self.persist_directory)
        except Exception as e:
            logger.critical(f"On creating db: {e}")

    def load_data(self) -> List[Document]:
        """ Loads recognized text data from disk """
        loader = DirectoryLoader(self.persist_directory, glob="**/*.txt", loader_cls=TextLoader)
        docs = loader.load()
        return docs

    def split_data(self, docs: List[Document], chunk_size: int = 800, chunk_overlap: int = 200) -> List[Document]:
        """ Split data into chunks"""
        splitter = CharacterTextSplitter(separator=".", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(docs)
        return chunks

    def append_files(self) -> int:
        """ Appends chunks to database """
        try:
            print("Documents loading...")
            docs = self.load_data()
            if len(docs) == 0:
                return 0
            chunks = self.split_data(docs)
            self.db.add_documents(chunks)
            print(f"Added {len(docs)} documents")
            logger.info(f"Added {len(chunks)} chunks to {self.collection_name}")
        except Exception as e:
            print(f"Error adding documents: {e}")
            logger.error(f"Failed to append files to {self.collection_name}: {e}")
            return -1

        # # deleting raw txt files
        # try:
        #     files = os.listdir(self.persist_directory)
        #     for file in files:
        #         if file.endswith(".txt"):
        #             os.remove(os.path.join(self.persist_directory, file))
        # except Exception as e:
        #     logger.error(f"Failed deleting files: {e}")
        #     return 1

        return 1
