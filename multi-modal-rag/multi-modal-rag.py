import streamlit as st

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_community.retrievers import BM25Retriever
from nltk.tokenize import word_tokenize
from langchain.retrievers import EnsembleRetriever

# https://github.com/NarimanN2/ollama-playground/blob/main/hybrid-retrieval-rag/hybrid_pdf_rag.py

