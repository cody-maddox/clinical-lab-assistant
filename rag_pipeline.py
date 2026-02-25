"""
rag_pipeline.py

Production pipeline for the Clinical Lab RAG system.
Loads the pre-built ChromaDB index and exposes route_query() for use by the Streamlit app.

This module does NOT build or rebuild the index — it assumes chroma_db/ already exists
and is populated. To rebuild, run the relevant cells in RAG.ipynb.
"""

import os
import logging
from pathlib import Path

import chromadb
from dotenv import load_dotenv

from llama_index.core import Settings, PromptTemplate, VectorStoreIndex
from llama_index.core.llms import ChatMessage
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# BASE_DIR is the directory this file lives in (Clinical_RAG/).
# Using __file__ makes paths work regardless of where you launch the app from.
BASE_DIR = Path(__file__).parent
CHROMA_DB_PATH = str(BASE_DIR / "chroma_db")

# ---------------------------------------------------------------------------
# API key + model settings
# ---------------------------------------------------------------------------

load_dotenv(BASE_DIR / ".env")

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-4o-mini")

# Suppress noisy httpx request logs from OpenAI calls
logging.getLogger("httpx").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Load ChromaDB index
# ---------------------------------------------------------------------------

persistent_client = chromadb.PersistentClient(CHROMA_DB_PATH)
collection = persistent_client.get_or_create_collection(name="lab_tests")

from llama_index.vector_stores.chroma import ChromaVectorStore
vector_store = ChromaVectorStore(chroma_collection=collection)

# from_vector_store loads the existing embeddings — no re-chunking or re-embedding
index = VectorStoreIndex.from_vector_store(vector_store)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

qa_template = PromptTemplate(
"""You are a patient education assistant providing accurate medical information.

Context information from trusted medical sources is listed below:

---------------------
{context_str}
---------------------

Important instructions:
- Answer ONLY based on the context provided above
- Use clear, patient-friendly language (avoid medical jargon when possible)
- If the answer isn't in the context, say "I don't have enough information to answer that accurately"
- When answering questions about procedures, reference ranges, or specimen requirements, include all relevant details — do not summarize or omit specifics
- Always cite which source(s) you're using
- Always end your response with the disclaimer: "This is educational information only. Please consult your healthcare provider for personalized medical advice."
- If asked for diagnosis or treatment decisions, redirect to consulting a healthcare provider

Question: {query_str}

Answer: """
)

# ---------------------------------------------------------------------------
# Source-specific query engines
# ---------------------------------------------------------------------------

medline_engine = index.as_query_engine(
    similarity_top_k=3,
    text_qa_template=qa_template,
    filters=MetadataFilters(filters=[ExactMatchFilter(key='source', value='medline')]),
    streaming=True
)
mayo_engine = index.as_query_engine(
    similarity_top_k=3,
    text_qa_template=qa_template,
    filters=MetadataFilters(filters=[ExactMatchFilter(key='source', value='mayo')]),
    streaming=True
)

# ---------------------------------------------------------------------------
# Intent classifier
# ---------------------------------------------------------------------------

CLASSIFIER_SYSTEM_PROMPT = """You are a query classifier for a medical laboratory test information assistant.
Classify the user's query into exactly one of these categories:

- in_scope: The query asks about laboratory tests — what they measure, why they are ordered, how to prepare, what results mean, reference ranges, specimen handling, tube types, or collection procedures.
- out_of_scope: The query is not related to laboratory tests.
- medical_emergency: The query describes an urgent medical situation, severe symptoms, or asks if the user is in danger.
- diagnosis_request: The query asks for a personal medical diagnosis or whether the user has a specific condition.

Respond with ONLY the category label. No explanation. No punctuation. Just the label."""

def classify_query(query: str) -> str:
    messages = [
        ChatMessage(role="system", content=CLASSIFIER_SYSTEM_PROMPT),
        ChatMessage(role="user", content=query)
    ]
    response = Settings.llm.chat(messages)
    return response.message.content.strip().lower()

# ---------------------------------------------------------------------------
# Source classifier
# ---------------------------------------------------------------------------

SOURCE_CLASSIFIER_PROMPT = """You are a routing classifier for a medical laboratory assistant with two source databases:
- patient_education: MedlinePlus articles written for patients covering explanations, purposes, and preparation
- lab_procedures: Mayo Clinic lab handbook with specimen collection protocols and reference values

Classify the query into exactly one of these categories:

- patient_education: Use when the question asks WHY a test is ordered, WHAT a test is or does conceptually, WHAT results mean for a patient, HOW TO PREPARE for a test, or compares tests at a conceptual level.
  Examples: "Why do I need a CBC?", "What does a high WBC indicate?", "Do I need to fast?", "What is the difference between BMP and CMP?"

- lab_procedures: Use when the question asks for specific collection requirements (tube type, volume, handling), specimen stability windows, specific numeric reference ranges by age or sex, or clinical diagnostic thresholds.
  Examples: "What tube type is required?", "What is the hemoglobin reference range for males 6-8 years?", "How long is the specimen stable?", "What HbA1c threshold does the ADA use?"

Respond with ONLY the category label. No explanation. No punctuation. Just the label."""

def classify_source(query: str) -> str:
    messages = [
        ChatMessage(role="system", content=SOURCE_CLASSIFIER_PROMPT),
        ChatMessage(role="user", content=query)
    ]
    response = Settings.llm.chat(messages)
    return response.message.content.strip().lower()

# ---------------------------------------------------------------------------
# Canned responses + router
# ---------------------------------------------------------------------------

CANNED_RESPONSES = {
    "out_of_scope": "I can only answer questions about laboratory tests. Please ask about a specific lab test.",
    "medical_emergency": "If you are experiencing a medical emergency, call 911 immediately. This assistant cannot help with urgent medical situations.",
    "diagnosis_request": "I'm not able to provide a personal medical diagnosis. Please consult your healthcare provider to discuss your specific results."
}

def route_query(query: str):
    category = classify_query(query)
    if category == "in_scope":
        source = classify_source(query)
        if source == "lab_procedures":
            return mayo_engine.query(query)
        else:
            return medline_engine.query(query)
    else:
        return CANNED_RESPONSES.get(category, CANNED_RESPONSES['out_of_scope'])
