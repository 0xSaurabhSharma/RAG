from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langcahin.vectorstore import FAISS
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain import PromptTemplate
# rate limit err
from typing import List
from rank_bm25 import BM250kapi
import fitz
import asyncio
import random
import textwrap
import numpy as np
from enum import Enum


def replace_t_with_space (list_of_documents):
    """
    Replaces all tab characters ('\t') with spaces in the page content of each document

    Args:
        list_of_documents: A list of document objects, each with a 'page_content' attribute.

    Returns:
        The modified list of documents with tab characters replaced by spaces.
    """

    for doc in list_of_documents:
        doc.page_content = doc.page_content.replace('\t',' ')
    return list_of_documents


def encode_pdf (path, chunk_size=1000, chunk_overlap=200):
    """
    Encodes a PDF book into a vector store using OpenAI embeddings.

    Args:
        path: The path to the PDF file.
        chunk_size: The desired size of each text chunk.
        chunk_overlap: The amount of overlap between consecutive chunks.

    Returns:
        A FAISS vector store containing the encoded book content.
    """

    loader = PyPDFLoader(path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size= chunk_size, chunk_overlap=chunk_overlap, length_function=len)
    split_docs = splitter.split_documents(docs)
    cleaned_split = replace_t_with_space(split_docs)

    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embedding = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    vectorstore = FAISS.from_documents()