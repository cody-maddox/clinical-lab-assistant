# RAG + Agent System — High-Level Workflow

This document outlines a **portfolio-level Retrieval-Augmented Generation (RAG) system with an agent layer**, emphasizing experimentation, evaluation, and production-oriented design rather than simple chatbot implementation.

Project follows completion of the book: Building LLMs for Production by Louis-Francois Bouchard, Louie Peters

Preferred tools to star: llamaindex, chromadb

---

# 1. Problem Definition & Dataset Design

**Goal:** Clearly define what success looks like before building.

Key decisions:
- Domain scope (healthcare)
    - Scope = create RAG model patient assistant for top 10-20 most common laboratory tests
    - Can answer questions using ground truth from source URLs (downloaded pdfs) from below.
    - These pdfs include: what is the test, what is it used for, why do I need this blood test, how to prepare for the test, risks, what do results mean, etc.
- Expected output format (summary, comparison, structured report)
- Grounding requirements (citations required?)
- Evaluation criteria

Outputs:
- Curated document corpus
- Evaluation dataset (“gold” questions)

> This stage determines downstream architecture decisions.

---

# 2. Data Ingestion & Preprocessing

**Goal:** Convert raw data into structured, retrievable knowledge.

Typical steps:
- Parse PDFs, webpages, or documents
    - Source 1 lab tests (general info): https://medlineplus.gov/lab-tests/
    - Source 2 (more specific info): https://www.mayocliniclabs.com/test-catalog
    - Top 10-20 most common lab tests will be pulled from these sources, downloaded as pdfs and used as ground truth for query responses
- Clean and normalize text
- Remove boilerplate content
- Extract metadata (author, section, date, topic)

### Experimentation Opportunities
- Parser selection
- Metadata enrichment strategies
- Structured vs unstructured ingestion

Insight:
Retrieval quality often depends more on preprocessing than model choice.

---

# 3. Chunking Strategy

**Goal:** Break documents into effective retrieval units.

Possible approaches:
- Fixed token windows
- Semantic chunking
- Sliding window overlap
- Section-aware chunking
- Hierarchical chunking

### Tradeoffs
- Smaller chunks → precise retrieval, less context
- Larger chunks → richer context, more noise

Evaluation focus:
- Retrieval recall
- Answer completeness

---

# 4. Embeddings & Index Design

**Goal:** Represent knowledge in vector space for retrieval.

### Embedding Experiments
- General-purpose vs domain-specific embeddings
- Different embedding dimensions
- Fine-tuned embeddings (advanced)

### Index Strategies
- Vector search
- Hybrid search (BM25 + vector)
- Metadata filtering
- Reranking models

Metrics:
- Recall@K
- Relevance scoring
- Latency

---

# 5. Retrieval Pipeline

**Goal:** Retrieve the most relevant context for each query.

Components:
- Query rewriting
- Multi-query retrieval
- Reranking
- Context compression

Extensions:
- Query expansion agent
- Adaptive Top-K selection
- Semantic filtering

---

# 6. Agent Layer (Decision & Control)

**Goal:** Introduce reasoning and adaptive workflow control.

Potential Agent responsibilities:
- Decide when retrieval is required
- Guardrails: Determine if query is inappropriate/dangerous?
    - Example: Am I going to die?
    - Stress that this is not medical advice and assistant is NOT replacement for a physician
- Reformulate weak queries
- Detect insufficient evidence
- Retry retrieval when needed
- Select tools dynamically

Typical agent roles:
- Planner agent
- Retrieval executor
- Verifier/evaluator
- Answer synthesizer

Key principle:
Agents improve **decision quality**, not just automation.

---

# 7. Generation & Grounded Answering

**Goal:** Produce answers supported by retrieved evidence.

Design considerations:
- Structured outputs
- Citation enforcement
- Uncertainty reporting
- Constrained prompting

Extensions:
- Reasoning traces
- Multi-document synthesis
- Comparative analysis

---

# 8. Evaluation Layer (Critical Differentiator)

**Goal:** Measure system performance objectively.

Evaluation targets:
- Groundedness / faithfulness
- Retrieval relevance
- Hallucination rate
- Completeness
- Consistency across runs

Methods:
- LLM-as-judge evaluation
- Embedding similarity scoring
- Human validation subsets

---

# 9. Reflection & Retry Loop (Advanced Agent Behavior)

**Goal:** Automatically improve weak responses.

System detects:
- Low grounding scores
- Missing citations
- Low confidence outputs

Actions:
- Retry retrieval
- Rewrite queries
- Request additional context
- Regenerate answers

This is where agentic systems provide real value.

---

# 10. Observability & Production Monitoring

**Goal:** Understand behavior and performance over time.

Tracked metrics:
- Latency
- Token usage / cost
- Retrieval success rate
- Retry frequency
- Failure patterns

Extensions:
- Experiment dashboards
- Cost vs accuracy analysis
- Performance monitoring

---

# Other extensions/thoughts

- Experiment with voice-to-text user query (Whisper package?)
- Deploy frontend web application (Streamlit?)
- Expand sources/type of questions assistant can answer
- Conversational memory
    - from llama_index.core.memory import ChatMemoryBuffer
- Feedback loop: let users rate answers to improve retrieval
- Advanced query engine with custom prompts and post-processing

    qa_prompt = PromptTemplate(
        """You are a patient education assistant providing accurate medical information.
        
Context information from trusted medical sources is below:
---------------------
{context_str}
---------------------

Important instructions:
- Answer ONLY based on the context provided above
- Use clear, patient-friendly language (avoid medical jargon when possible)
- If the answer isn't in the context, say "I don't have enough information to answer that accurately"
- Always cite which source(s) you're using
- Include this disclaimer: "This is educational information only. Please consult your healthcare provider for personalized medical advice."
- If asked for diagnosis or treatment decisions, redirect to consulting a healthcare provider

Question: {query_str}

Answer: """

- Make chat interactive 
- Stream responses

# Notes

- Use OpenAI API: GPT-4o-mini
- Start small w/ 3 tests to validate, then expand to 15-20
    - 3 tests lets you see if model pulls from Mayo for technical Qs and Medline for general Qs
    - Use metadata tagging
- Name files similarly for both Mayo and Medline directories
    - Ex: cbc.pdf in both directories
- Medline pdfs have long References section that should be removed when parsing (no value & will pollute vector space)
    - Split on "References\n", keep everything before it
- Mayo pdfs: Strip repeating page headers and the administrative footer section
    - Split on "Fees & Codes", keep everything before it.
    - Also strip the repeating page headers with a regex or string match on the "Test Definition:..." / "Document generated..." / "Page X of Y" pattern.
- Index strategies
    - Return/experiment w/ reranking
- Baseline test set experiment: 12 curated Qs (2 for each test, 1 straightforward & 1 specific) w/ expected answers from test
    - Test 1: visual inspection & scoring of responses
    - Test 2: llamaindex CorrectnessEvaluator w/ more powerful LLM (gpt-4o)
- Agents scope:
    - Decide if a Q needs retrieval, if out of scope, or if it needs medical disclaimer guardrail


# Tier 1 — Core RAG (build this first, make it work well):

- Data ingestion from your MedlinePlus/Mayo Clinic PDFs
- Chunking strategy
- Embeddings + ChromaDB vector store
- Basic retrieval + LLM generation with citations
- Custom prompt template (you already have a good draft in your notes)

# Tier 2 — What makes it portfolio-worthy:

- Evaluation layer (groundedness, retrieval relevance) — this is what separates a toy project from a real one
- Guardrails for medical queries
- The "this is not medical advice" safety layer

# Tier 3 — Nice-to-have extensions:

- Agent layer, reflection/retry loops, Streamlit frontend, Whisper voice input, conversational memory





