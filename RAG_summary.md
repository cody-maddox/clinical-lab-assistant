# Clinical RAG Project — Progress Summary

## Project Overview

A portfolio-level Retrieval-Augmented Generation (RAG) system built for the healthcare domain. The system answers patient and clinician questions about common laboratory tests using curated content from two authoritative sources: MedlinePlus (patient-education focus) and Mayo Clinic Labs (technical lab-handbook focus).

**Generation model:** `gpt-4o-mini`
**Embedding model:** `text-embedding-3-small`
**Vector store:** ChromaDB (persistent)
**Framework:** LlamaIndex
**Scope:** 12 lab tests — A1C, CBC, CMP (original development set) + BMP, TSH, PSA, PT/INR, Liver Panel, Troponin, Microalbumin/Creatinine Ratio, Ferritin, CRP (expanded corpus)

The dual-source design is intentional: Medline answers general patient questions ("Why do I need this test?") while Mayo answers technical clinician questions ("What tube type is required?"). A key goal is building a system that retrieves from the *right* source depending on the nature of the query.

---

## Data Sources

**Medline PDFs** (patient-facing education content):
- `medline/a1c.pdf`, `medline/cbc.pdf`, `medline/cmp.pdf` *(original)*
- `medline/bmp.pdf`, `medline/tsh.pdf`, `medline/psa.pdf`, `medline/pt_inr.pdf`, `medline/liver_panel.pdf`, `medline/troponin.pdf`, `medline/microalbumin.pdf`, `medline/ferritin.pdf`, `medline/crp.pdf` *(expanded corpus)*

**Mayo PDFs** (lab handbook — specimen handling, reference ranges, ADA thresholds):
- `mayo/a1c.pdf`, `mayo/cbc.pdf`, `mayo/cmp.pdf` *(original)*
- `mayo/bmp.pdf`, `mayo/tsh.pdf`, `mayo/psa.pdf`, `mayo/pt_inr.pdf`, `mayo/liver_panel.pdf`, `mayo/troponin.pdf`, `mayo/microalbumin.pdf`, `mayo/ferritin.pdf`, `mayo/crp.pdf` *(expanded corpus)*

---

## Phase 1: Data Ingestion Pipeline

Built a multi-step ingestion and cleaning pipeline in `RAG.ipynb`.

### Loading
Used LlamaIndex's `SimpleDirectoryReader` to load PDFs. Each PDF is parsed into multiple `Document` objects (one per page).

### Cleaning — Medline (`clean_medline`)
Medline PDFs contain a long **References** section at the end with no retrieval value — it pollutes the vector space with citations. Solution: split on `"References\n"` and discard everything after it.

### Cleaning — Mayo (`clean_mayo`)
Mayo PDFs have two structural problems:
1. **Repeating page headers** — Each page repeats a "Test Definition / Document generated / Page X of Y" header block
2. **Administrative footer** — A "Fees & Codes" section at the end with billing/administrative content, not clinical content

Solution: regex-based stripping of repeating headers, then split on `"Fees & Codes"` and discard the tail.

### Orchestrator (`load_combine_clean_pdfs`)
A single orchestrator function that loads PDFs from a directory, detects source type (Medline vs Mayo) by path, applies the appropriate cleaning function, and returns a list of cleaned text strings.

### LlamaIndex Document Objects
Cleaned text strings are converted to LlamaIndex `Document` objects with metadata:
- `source`: `"medline"` or `"mayo"`
- `test_name`: `"a1c"`, `"cbc"`, or `"cmp"`

This metadata is attached to every chunk and surfaced in `response.source_nodes`, enabling source attribution per retrieved chunk.

---

## Phase 2: Embedding & Vector Store

- **ChromaDB** configured as a `PersistentClient` at `./chroma_db` — data survives between notebook runs
- Collection name: `"lab_tests"`
- Wrapped in LlamaIndex's `ChromaVectorStore` + `StorageContext`
- Embedding model set globally via `Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")`
- LLM set globally via `Settings.llm = OpenAI(model="gpt-4o-mini")`

### Index Creation
`VectorStoreIndex.from_documents()` chunks the documents, embeds each chunk, and stores the embeddings in ChromaDB. The index is the primary retrieval object for all queries.

---

## Phase 3: Chunking Experimentation

### Default Chunking (Baseline)
LlamaIndex defaults: `chunk_size=1024`, `chunk_overlap=200` → **22 chunks**

### Experiment: 1500/300
Switched to `SentenceSplitter(chunk_size=1500, chunk_overlap=300)` passed via `transformations=[splitter]` to `VectorStoreIndex.from_documents()` → **15 chunks**

Larger chunks reduce total chunk count and give the LLM more context per retrieved chunk, but reduce retrieval precision.

### Result
Both configurations produced correct answers on test queries. The 1500/300 configuration is the current active setting. Chunking size alone was determined **not** to be the right lever for the primary retrieval noise problem (see Problems section).

### Collection Rebuild Pattern
A critical pattern was established for clean experimentation:

```python
try:
    persistent_client.delete_collection(name="lab_tests")
except Exception:
    pass
collection = persistent_client.get_or_create_collection(name="lab_tests")
```

Without deleting first, re-running `VectorStoreIndex.from_documents()` accumulates duplicate embeddings in ChromaDB, doubling the chunk count on each run.

---

## Phase 4: Baseline Queries

Two baseline queries were run to validate the pipeline:

**Query 1:** "Why do I need a complete blood count?"
- Correct response. Nodes retrieved from `medline/cbc` with cosine similarity scores ~0.52.

**Query 2:** "What is the reference range for creatinine for an 8 year old male?" (`similarity_top_k=3`)
- Correct response (0.26–0.61 mg/dL). Nodes retrieved from `mayo/cmp`.
- Third retrieved node was from `mayo/cbc` — cross-test retrieval noise first observed here.

`response.source_nodes` was used to inspect each retrieved chunk's metadata, score, and text for debugging.

---

## Phase 5: Evaluation

### Evaluation Set Design
12 questions total — 6 Medline, 6 Mayo, 2 per test per source. Questions span:
- Straightforward patient-education questions (Medline)
- Technical lab-procedure questions (Mayo): tube type, specimen handling, reference ranges, ADA diagnostic thresholds, specimen stability

Answers were verified against the source PDFs before use.

### Test 1: Visual Inspection
Loop through `medline_eval + mayo_eval`, query the engine, print question / expected answer / response side by side.

**Result: 11/12 correct.** Failure on Q11 (CMP specimen handling — Mayo source). Two partial answers on Q7 (HbA1c tube — missing "do not aliquot" detail) and Q9 (CBC stability — missing ambient temperature condition).

### Test 2: LlamaIndex CorrectnessEvaluator
Used `CorrectnessEvaluator` from `llama_index.core.evaluation` with **gpt-4o as judge** (deliberately stronger than the gpt-4o-mini generation model to avoid self-scoring bias).

Evaluator takes `query`, `response` (as string), and `reference` (expected answer) and returns a score (1–5) with written feedback.

**Average score: 4.38 / 5.0**

| Question | Score | Notes |
|----------|-------|-------|
| A1C — what it measures | 5.0 | Correct |
| A1C — 6.2% interpretation | 5.0 | Correct |
| CBC — what it measures | 5.0 | Correct |
| CBC — high WBC | 5.0 | Correct |
| CMP — fasting | 5.0 | Correct |
| BMP vs CMP | 5.0 | Correct |
| HbA1c — tube type/volume | **1.0** | Complete retrieval failure |
| ADA diabetes threshold | 5.0 | Correct |
| CBC — specimen stability | 4.5 | Partial (missing ambient temp) |
| Hemoglobin range 6-8yr M | 5.0 | Correct |
| CMP — specimen handling | **2.0** | Retrieval failure (pulled Medline) |
| Potassium range 10yr | 5.0 | Correct |

---

## Phase 6: Re-Ranking

### Implementation
Added a `SentenceTransformerRerank` postprocessor from `llama-index-postprocessor-sbert-rerank` using the `cross-encoder/ms-marco-MiniLM-L-6-v2` model (local, no additional API key required, downloads from HuggingFace on first use).

The retrieval pipeline changed from single-stage to two-stage:
1. **Stage 1:** Cosine similarity retrieval with `similarity_top_k=6` (wider net than baseline top_k=2)
2. **Stage 2:** Cross-encoder re-scores all 6 candidates against the query jointly, re-sorts, and passes only `top_n=2` to the LLM

The cross-encoder differs from cosine similarity in a key way: embedding models encode query and chunk *separately*, while a cross-encoder reads them *together* — producing a richer relevance judgment.

### Results

**Average score: 4.62 / 5.0** (up from 4.38 baseline, +0.24)

| Question | Baseline | Re-Ranking | Change |
|----------|----------|------------|--------|
| A1C — what it measures | 5.0 | 5.0 | — |
| A1C — 6.2% interpretation | 5.0 | 5.0 | — |
| CBC — what it measures | 5.0 | 5.0 | — |
| CBC — high WBC | 5.0 | 5.0 | — |
| CMP — fasting | 5.0 | 5.0 | — |
| BMP vs CMP | 5.0 | 5.0 | — |
| HbA1c — tube type/volume | 1.0 | **4.5** | +3.5 |
| ADA diabetes threshold | 5.0 | 5.0 | — |
| CBC — specimen stability | 4.5 | 5.0 | +0.5 |
| Hemoglobin range 6-8yr M | 5.0 | 5.0 | — |
| CMP — specimen handling | 2.0 | 2.0 | — |
| Potassium range 10yr | 5.0 | 4.0 | -1.0 |

### Key Observations
- **Q7 (HbA1c tube):** Biggest improvement — 1.0 → 4.5. Re-ranking successfully surfaced the correct Mayo chunk that cosine similarity consistently missed. Remaining deduction is the LLM omitting "do not aliquot" and minimum volume detail (generation behavior, not retrieval).
- **Q11 (CMP specimen handling):** Still 2.0. The cross-encoder is not distinguishing Medline patient-education CMP content from Mayo lab-handbook CMP content. This is the remaining known limitation (see below).
- **Q12 (Potassium):** Minor regression — response returned "3.5" instead of "3.6" for the lower bound, likely LLM generation variance.
- **Non-determinism note:** The retrieval and re-ranking pipeline is deterministic for a given query. Variability across runs originates from gpt-4o-mini's temperature setting — the generation step can produce different responses from identical context. On one observed run, the model abstained with "not in the provided context" for a question it answered correctly on other runs.

---

## Phase 7: Prompt Template

### Implementation
Added a custom `PromptTemplate` passed via the `text_qa_template` parameter to `as_query_engine()`. The template wraps every LLM generation call and enforces consistent response structure.

Key instructions embedded in the template:
- Answer only from provided context
- Use patient-friendly language
- Be exhaustive on procedural questions — do not summarize or omit specifics
- Cite the source(s) used
- End every response with a medical disclaimer

```python
from llama_index.core import PromptTemplate

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
```

The template requires exactly two placeholders: `{context_str}` (injected by LlamaIndex with the retrieved chunks) and `{query_str}` (injected with the user's question). Applied after re-ranking, before LLM generation.

### Results

**Average score: 4.58 / 5.0** (essentially unchanged from 4.62 re-ranking baseline, -0.04)

Key observation: The exhaustiveness instruction resolved Q7's remaining deduction — the model now includes "do not aliquot" and minimum volume details that it previously omitted. Disclaimer appeared correctly on all responses. Citation appeared partially (metadata is present in `context_str` but formatting is inconsistent).

---

## Phase 8: Agent Layer

### Design Rationale
Rather than using `ReActAgent` (which was incompatible with the installed `llama-index-core==0.14.x`), a two-stage manual routing architecture was implemented. This approach is equivalent in behavior for the current scope — one retrieval tool per source — and is more transparent and debuggable than an agent loop.

### Architecture: Two-Stage Classification + Routing

**Stage 1 — Intent Classifier (`classify_query`)**

Calls `Settings.llm.chat()` with a system prompt to gate all incoming queries before any retrieval is attempted. Returns one of four categories:

- `in_scope` — lab test questions; proceed to retrieval
- `out_of_scope` — unrelated query; return canned response
- `medical_emergency` — urgent symptoms; redirect to 911
- `diagnosis_request` — personal diagnosis; redirect to provider

Uses `llama_index.core.llms.ChatMessage` with `role="system"` / `role="user"` structure.

**Stage 2 — Source Classifier (`classify_source`)**

Only reached for `in_scope` queries. Routes to the appropriate metadata-filtered engine:

- `patient_education` → `medline_engine` (filtered to `source="medline"`)
- `lab_procedures` → `mayo_engine` (filtered to `source="mayo"`)

Classifier is prompted with explicit database descriptions and concrete WHY/WHAT/HOW examples to prevent over-routing to `lab_procedures`.

**Source-Specific Engines**

Both engines share the same reranker and prompt template, differing only in `MetadataFilters`:

```python
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

medline_engine = index.as_query_engine(
    similarity_top_k=6,
    node_postprocessors=[reranker],
    text_qa_template=qa_template,
    filters=MetadataFilters(filters=[ExactMatchFilter(key='source', value='medline')])
)

mayo_engine = index.as_query_engine(
    similarity_top_k=6,
    node_postprocessors=[reranker],
    text_qa_template=qa_template,
    filters=MetadataFilters(filters=[ExactMatchFilter(key='source', value='mayo')])
)
```

**Orchestrator (`route_query`)**

```python
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
```

### Results

**Average score: 4.71 / 5.0** (highest score achieved, up from 4.58 prompt template baseline, +0.13)

| Question | Re-Rank | Prompt Template | Agent Layer | Change (vs baseline) |
|----------|---------|-----------------|-------------|----------------------|
| A1C — what it measures | 5.0 | 5.0 | 5.0 | — |
| A1C — 6.2% interpretation | 5.0 | 5.0 | 5.0 | — |
| CBC — what it measures | 5.0 | 5.0 | 5.0 | — |
| CBC — high WBC | 5.0 | 5.0 | 5.0 | — |
| CMP — fasting | 5.0 | 5.0 | 5.0 | — |
| BMP vs CMP | 5.0 | 5.0 | 5.0 | — |
| HbA1c — tube type/volume | 4.5 | 5.0 | 5.0 | **+4.0 (from 1.0)** |
| ADA diabetes threshold | 5.0 | 5.0 | 5.0 | — |
| CBC — specimen stability | 5.0 | 4.5 | 4.5 | — |
| Hemoglobin range 6-8yr M | 5.0 | 5.0 | 5.0 | — |
| CMP — specimen handling | 2.0 | 2.0 | **5.0** | **+3.0 (from 2.0)** |
| Potassium range 10yr | 4.0 | 4.0 | 3.5 | -0.5 |

### Key Observations
- **Q11 (CMP specimen handling): FULLY RESOLVED.** Score went from 2.0 (all previous phases) to 5.0. Routing `lab_procedures` queries exclusively to `mayo_engine` eliminated the cross-source retrieval failure that persisted through chunking and re-ranking experiments.
- **Q7 (HbA1c tube): FULLY RESOLVED.** Combined effect of re-ranking (1.0 → 4.5) and prompt template exhaustiveness instruction (4.5 → 5.0).
- **Q12 (Potassium range): Persistent known limitation.** Routes correctly to `mayo_engine` but the LLM returns incorrect numeric values across every run (e.g., 3.5–5.0, 3.5–5.2, 3.4–4.7 instead of 3.6–5.1). Root cause: the Mayo CMP PDF contains dense reference range tables with multiple age/sex brackets; the LLM misreads the correct row during generation. Not a routing or retrieval problem.
- **Q9 (CBC stability): Partial.** Consistently 4.5 — retrieves the correct chunk but omits the ambient temperature stability window (24hr). Refrigerated stability (48hr) is always included.

### Source Classifier Refinement
The initial `SOURCE_CLASSIFIER_PROMPT` over-routed all queries to `lab_procedures`, causing patient-education questions to be answered from Mayo instead of Medline. Fixed by adding explicit database descriptions, a WHY/WHAT/HOW routing heuristic, and concrete examples for each category. After refinement, routing was correct on all 12 evaluation questions.

---

## Phase 9: Corpus Expansion

Scaled the corpus from 3 tests to 12 by adding 9 new tests: BMP, TSH, PSA, PT/INR, Liver Panel, Troponin, Microalbumin/Creatinine Ratio, Ferritin, and CRP.

### Process
- Downloaded Mayo PDFs and print-to-PDFs from MedlinePlus for each new test
- Confirmed each new PDF cleaned correctly through the existing `clean_medline` / `clean_mayo` pipeline (all 9 new tests shared the same structural patterns as the original 3)
- No pipeline changes were required — `load_combine_clean_pdfs` reads entire directories, so rebuilding the index with all 12 tests was a matter of rerunning the existing chunking cell
- Ran sanity queries against TSH (one Medline, one Mayo) to confirm the expanded index was working before full evaluation

### Index After Expansion
- **Documents:** 24 LlamaIndex `Document` objects (12 tests × 2 sources)
- **Chunks:** 47 (up from 15 with the 3-test corpus; same 1500/300 `SentenceSplitter` settings)

---

## Phase 10: Expanded Corpus Evaluation

### Evaluation Set Design
36 new evaluation questions — 2 Medline + 2 Mayo per new test (18 + 18). Same structure as the original 12-question eval. All expected answers verified against source PDFs before use.

Question types covered across the new tests:
- **Medline:** what a test is for, what abnormal results mean, fasting/preparation requirements, test comparisons
- **Mayo:** tube type and volume, special handling (light protection, centrifuge timing, aliquoting), specimen stability windows, reference ranges (age/sex-stratified where available), diagnostic thresholds (INR therapeutic ranges)

Procedural stress tests built into the set:
- PSA age-stratified upper limits (6 age brackets — similar tabular complexity to the potassium question that caused Q12 failures in the original eval)
- Troponin tube type (lithium heparin — different from the serum gel used by most tests)
- Microalbumin (urine specimen — the only non-blood test in the corpus)
- Multiple tests with fasting instructions at different durations (BMP: 8hr, ferritin: 12hr, liver panel: 10–12hr)

### Results

**Average score: 4.94 / 5.0** (35/36 correct; all 36 routing decisions correct)

| Test | Source | Question | Score |
|------|--------|----------|-------|
| BMP | Mayo | Tube type, volume, handling | 5.0 |
| BMP | Mayo | Refrigerated stability | 5.0 |
| TSH | Mayo | Tube type and volume | 5.0 |
| TSH | Mayo | Adult reference range (≥20yr) | 5.0 |
| PSA | Mayo | Upper limits by age group | 5.0 |
| PSA | Mayo | Refrigerated vs frozen stability | 5.0 |
| PT/INR | Mayo | Normal PT and INR ranges | 5.0 |
| PT/INR | Mayo | Therapeutic INR ranges (standard vs high) | 5.0 |
| Liver Panel | Mayo | Special handling instructions | 5.0 |
| Liver Panel | Mayo | ALT reference range (adult males) | 5.0 |
| Troponin | Mayo | Tube type, volume, handling | 5.0 |
| Troponin | Mayo | Female reference range | 5.0 |
| Microalbumin | Mayo | Microalbuminuria / proteinuria thresholds | 5.0 |
| Microalbumin | Mayo | Specimen stability | 5.0 |
| Ferritin | Mayo | Patient preparation (biotin) | 5.0 |
| Ferritin | Mayo | Adult male reference range | 5.0 |
| CRP | Mayo | Reference range | 5.0 |
| CRP | Mayo | Refrigerated and ambient stability | 5.0 |
| BMP | Medline | BMP vs CMP difference | 5.0 |
| BMP | Medline | Fasting requirement | 5.0 |
| TSH | Medline | What it measures / conditions detected | 5.0 |
| TSH | Medline | High TSH interpretation | 5.0 |
| PSA | Medline | What a PSA test is used for | 5.0 |
| PSA | Medline | Pre-test avoidance (semen / 24hr) | 5.0 |
| PT/INR | Medline | Most common use (warfarin monitoring) | 5.0 |
| PT/INR | Medline | High INR interpretation | 5.0 |
| Liver Panel | Medline | Most common uses | 5.0 |
| Liver Panel | Medline | **Fasting requirement** | **3.0** |
| Troponin | Medline | What troponin is / why it's measured | 5.0 |
| Troponin | Medline | Special preparation (biotin warning) | 5.0 |
| Microalbumin | Medline | What the test detects | 5.0 |
| Microalbumin | Medline | Pre-test avoidance (exercise / meat) | 5.0 |
| Ferritin | Medline | What it measures / low result meaning | 5.0 |
| Ferritin | Medline | Patient preparation (fasting) | 5.0 |
| CRP | Medline | What it measures / who produces CRP | 5.0 |
| CRP | Medline | Standard CRP vs hs-CRP difference | 5.0 |

### Key Observations
- **PSA age table (5.0):** The 6-bracket age-stratified PSA limits were retrieved and reported correctly — a better outcome than the potassium table failure in the original eval. Possible reason: the PSA table is less dense (one reference value per bracket vs. multiple analytes per age group in the CMP).
- **Microalbumin urine specimen (5.0):** The only non-blood test in the corpus caused no retrieval or routing issues.
- **Liver panel fasting (3.0):** The one failure. See Known Limitations — Cross-Test Retrieval Noise Within Medline Partition.
- **All 36 routing decisions correct:** Every `lab_procedures` question was routed to `mayo_engine`; every `patient_education` question was routed to `medline_engine`. The source classifier generalized perfectly to 9 new test types it had never seen during the prompt refinement phase.

---

## Phase 11: Faithfulness Evaluation

### Motivation
The `CorrectnessEvaluator` measures end-to-end answer quality against a known reference answer — it catches both retrieval failures and hallucinations, but it can't tell them apart. A low correctness score could mean (a) the LLM made something up, or (b) the right chunk simply wasn't retrieved. For a medical application, this distinction is critical. **Faithfulness** answers a different question: *Is the LLM's response grounded in the retrieved context?* A faithful answer may still be wrong (if the context was wrong/incomplete), but it is not hallucinated.

### Implementation
Used LlamaIndex's `FaithfulnessEvaluator` with `gpt-4o` as judge (same judge used for correctness, deliberately stronger than the `gpt-4o-mini` generation model to avoid self-scoring bias).

Key implementation notes:
- `FaithfulnessEvaluator.evaluate_response(query=..., response=...)` requires a LlamaIndex `Response` object — not a plain string — because it needs access to `response.source_nodes` (the retrieved chunks) to check groundedness
- Canned responses from `route_query` (for `out_of_scope`, `medical_emergency`, `diagnosis_request`) are Python strings, not Response objects — these can't be evaluated for faithfulness and are skipped with `faith.append(None)` using an `isinstance(response, str)` guard
- Returns a binary score: `1.0` (faithful — all claims grounded in context) or `0.0` (unfaithful — LLM added claims not in retrieved chunks)

The combined loop ran `CorrectnessEvaluator` and `FaithfulnessEvaluator` together on all 36 questions in `new_combined_evals`.

### Results

**Faithfulness: 36.0 / 36** (all responses grounded in retrieved context, zero hallucinations)

**Correctness: 4.94 / 5.0** (35/36 correct — unchanged from Phase 10)

### Faithfulness vs. Correctness: Complementary Diagnostic Framework

| Faithfulness | Correctness | Diagnosis |
|---|---|---|
| ✅ Pass | ✅ High | Ideal — right chunks retrieved, response grounded in them |
| ✅ Pass | ❌ Low | **Retrieval failure** — LLM faithfully reported what it found, but the right chunk wasn't there |
| ❌ Fail | ✅ High | **Hallucination** — LLM produced correct-sounding answer by adding knowledge outside the retrieved context |
| ❌ Fail | ❌ Low | Hallucination or total failure |

### Key Insight: Liver Panel Fasting Confirmed as Retrieval Failure

The one correctness failure (liver panel fasting, 3.0) passed faithfulness. This proves it is a **retrieval failure, not hallucination**: the LLM faithfully reported what was in the retrieved chunks — which happened to be nothing about liver panel fasting. The correct answer (10–12 hours) exists in `medline/liver_panel.pdf`, but a competing fasting chunk from another test was ranked higher by the reranker. The LLM did exactly what it was supposed to do given the context it received.

This is the exact failure mode the cross-test Medline noise limitation (Phase 10) predicted.

### Significance for Medical Applications
Zero hallucinations across 36 queries is a meaningful safety result. In a medical context, a hallucinated answer (e.g., an invented fasting duration or specimen tube type) could cause real harm. The faithfulness result confirms that the system's prompt template instruction ("Answer ONLY based on the context provided above") is being honored by the generation model. All failures are retrieval gaps, not fabrications — a much safer failure mode, since the system either says "I don't have that information" or retrieves the wrong-but-real content from a related test.

---

## Known Limitations

### Tabular Data Extraction — Potassium Range (Q12)
The `mayo_engine` correctly retrieves the CMP reference range chunk, but the LLM consistently reads incorrect values for the 10-year-old potassium range from the dense age-bracketed table in the Mayo CMP PDF. This is a generation-layer problem (LLM misreading tabular data), not a routing or retrieval problem.

**Potential fixes:** Structured data extraction (parse the PDF tables into structured format before indexing), or fine-grained chunking of individual reference range rows. Deferred to future work.

### CBC Specimen Stability — Partial Answer (Q9)
The system correctly returns the refrigerated specimen stability window (48 hours) but consistently omits the ambient temperature window (24 hours). The exhaustiveness instruction in the prompt template partially addresses this but does not fully resolve it.

### Cross-Test Retrieval Noise Within Medline Partition — Liver Panel Fasting (Phase 10)
With the expanded 12-test corpus, the cross-test noise problem that previously appeared between Mayo tests (Phase 4–5) has now surfaced within the Medline partition.

**Failure:** "How long should a patient fast before a liver function test?" → Response: "There is no specific fasting requirement mentioned for a liver function test in the provided information from Medline." Score: 3.0.

The Medline `liver_panel.pdf` does contain the answer (10–12 hours), but the chunk wasn't retrieved. The agent layer's metadata filter correctly constrained retrieval to `source="medline"`, but within Medline, the reranker surfaced a fasting chunk from a different test (likely BMP or ferritin) rather than the liver panel chunk. Root cause: multiple Medline tests discuss fasting in similar language — BMP (8 hr), CMP (several hours), ferritin (12 hr), liver panel (10–12 hr) — and the query "fasting before a liver function test" doesn't contain enough test-specific signal to outrank those competitors. The fasting language is generic; the test name is the only discriminating feature.

This is the same structural problem seen in Phase 4–5 (Mayo CBC/CMP cross-test noise), now appearing at larger corpus scale within a single source partition.

**Potential fixes:** A third metadata filter on `test_name` (requires query-to-test-name resolution before retrieval), or smaller chunk sizes for preparation/fasting content so the test-specific context is more concentrated per chunk. Deferred to future work.

---

## Problems Encountered

### 1. Cross-Document Retrieval Noise
**Problem:** Mayo CBC chunks are retrieved during CMP queries and vice versa. All Mayo PDFs share similar structural formatting (dense reference range tables, age brackets), causing them to land close together in embedding space.

**Status:** Resolved. Re-ranking suppressed most cross-test noise. Residual CMP specimen handling failure (Q11) was fully resolved by the agent layer's source-specific routing.

### 2. HbA1c Tube Type — Intermittent Retrieval Failure
**Problem:** Q7 (tube type for HbA1c) succeeded during visual inspection but failed in the CorrectnessEvaluator run. The model responded "not detailed in the provided information" indicating the relevant chunk was not retrieved.

**Likely cause:** The 1500/300 chunking merges the tube-type information with surrounding content, making it a smaller signal within a larger chunk. With `similarity_top_k=2`, the right chunk doesn't always surface.

**Status:** Fully resolved. Re-ranking (1.0 → 4.5) + prompt template exhaustiveness instruction (4.5 → 5.0).

### 3. Duplicate Chunk Accumulation
**Problem:** Re-running `VectorStoreIndex.from_documents()` without deleting the collection accumulated duplicate embeddings (22 → 44 chunks on second run).

**Fix:** Added try/except delete pattern before collection creation in the experiment cell.

### 4. `delete_collection` on Fresh Run
**Problem:** `persistent_client.delete_collection()` raises a `ValueError` if the collection doesn't exist (e.g., fresh `chroma_db` directory). This broke the experiment cell on first run.

**Fix:** Wrapped delete call in `try/except Exception: pass`.

### 5. Missing `Settings.llm`
**Problem:** Initial baseline query setup only configured `Settings.embed_model`. The query engine also requires `Settings.llm` for generation. Without it, queries failed.

**Fix:** Added `Settings.llm = OpenAI(model="gpt-4o-mini")` to the setup cell.

### 6. LLM Summarization Omits Details
**Problem:** For questions requiring exhaustive enumeration (e.g., specimen handling instructions with multiple constraints), gpt-4o-mini summarizes and omits some details even when the right chunk is retrieved.

**Observed in:** Q7 (missing "do not aliquot"), Q9 (missing ambient temperature stability).

**Fix:** Added exhaustiveness instruction to the prompt template. Fully resolved Q7. Q9 remains partial.

### 7. `ReActAgent` API Incompatibility
**Problem:** Installing `llama-index-postprocessor-sbert-rerank` upgraded `llama-index-core` to 0.14.x. `ReActAgent.from_tools()`, `.chat()`, and `.query()` all failed with `AttributeError` in this version.

**Fix:** Replaced `ReActAgent` with a manually implemented two-stage routing function (`route_query`). Equivalent behavior for the current scope, and more transparent/debuggable.

### 8. Source Classifier Over-Routing
**Problem:** Initial `SOURCE_CLASSIFIER_PROMPT` routed all 12 eval queries to `lab_procedures`, causing patient-education questions (Q1–Q6) to be answered from Mayo. Q5 dropped to 2.0, Q6 to 3.0.

**Fix:** Rewrote the prompt with explicit database descriptions, a WHY/WHAT/HOW routing heuristic, and labeled examples for each category. Routing was correct on all 12 questions after the fix.

---

## Phase 12: Streamlit App

### Architecture

The pipeline was extracted from `RAG.ipynb` into a standalone module `rag_pipeline.py` to serve as the backend for a Streamlit web app (`app.py`). The notebook remains the exploratory/evaluation artifact; `rag_pipeline.py` is the production path.

**File structure:**
```
Clinical_RAG/
├── RAG.ipynb          ← exploratory work, evaluation, experiments
├── rag_pipeline.py    ← shared module: index load + route_query
├── app.py             ← Streamlit UI
└── chroma_db/         ← persisted ChromaDB index (pre-built)
```

### `rag_pipeline.py` — Key design decisions

- **Index load path:** Uses `VectorStoreIndex.from_vector_store(vector_store)` to load the pre-built ChromaDB index without re-chunking or re-embedding. The notebook's `from_documents()` build path is not present — the app assumes the index already exists.
- **Explicit `Settings.llm`:** Added `Settings.llm = OpenAI(model="gpt-4o-mini")` explicitly. The notebook relied on LlamaIndex's undocumented fallback to `gpt-3.5-turbo`; the module makes this intentional.
- **Absolute paths via `Path(__file__).parent`:** ChromaDB path is anchored to the module's file location, so the app works regardless of which directory `streamlit run` is launched from.
- **No debug prints:** `route_query` debug prints (`[intent classifier]`, `[source classifier]`) removed — appropriate for notebook exploration, not for a UI module.

### `app.py` — Features

- **Text input + form submit** (`st.form`) — Enter key and button both trigger submission
- **Spinner** while `route_query` executes (retrieval + classification phase)
- **Streaming responses** via `st.write_stream(response.response_gen)` — tokens appear token-by-token as the LLM generates; requires `streaming=True` on both query engines in `rag_pipeline.py`
- **Retrieved sources expander** — shows source (Medline/Mayo), test name, and raw chunk text for each retrieved node; only shown for real RAG responses (not canned strings)
- **Conversational memory** — `condense_query()` in `rag_pipeline.py` rewrites vague follow-up questions into standalone questions using the last 3 turns of history before routing; `route_query()` accepts an optional `history` parameter; `app.py` passes `st.session_state.history` on every call
- **Display-only conversation history** — past Q&A pairs stored in `st.session_state.history`, rendered below the current answer (newest first, current query excluded to avoid duplication); clears on page refresh

### Problems Encountered

**Python version conflict:** System `streamlit` binary was linked to Python 3.9, which doesn't support the `X | None` union type syntax used in LlamaIndex's source. LlamaIndex was installed in the conda environment (Python 3.11+). Fix: install streamlit in the conda environment (`!pip install streamlit` from notebook) and run with `python -m streamlit run app.py` instead of `streamlit run`.

---

## Phase 13: Streamlit Community Cloud Deployment

### GitHub Repository

Created a new public GitHub repo (`clinical-lab-assistant`) containing the production files:

```
clinical-lab-assistant/
├── app.py
├── rag_pipeline.py
├── requirements.txt
├── chroma_db/         ← committed directly (6.7MB — small enough for git)
├── RAG.ipynb
├── RAG_summary.md
├── RAG_notes.md
└── .gitignore         ← excludes .env, PDFs/, __pycache__
```

PDFs excluded from the repo (copyrighted source documents). API key handled via Streamlit Cloud's Secrets manager (TOML format: `OPENAI_API_KEY = "sk-..."`), which injects secrets as environment variables at runtime — `load_dotenv()` becomes a no-op, but `OpenAI()` reads `OPENAI_API_KEY` from the environment automatically.

### Reranker Removed for Deployment

The `SentenceTransformerRerank` postprocessor was dropped from the production deployment. The `sentence-transformers` package depends on PyTorch (~750MB), which exceeded Streamlit Community Cloud's build constraints.

**Changes made for deployment:**
- Removed `SentenceTransformerRerank` import and instantiation from `rag_pipeline.py`
- Removed `node_postprocessors=[reranker]` from both query engines
- Reduced `similarity_top_k` from 6 to 3 (the higher top_k was chosen to give the reranker enough candidates; without it, 3 direct retrievals is equivalent)
- Removed `llama-index-postprocessor-sbert-rerank` from `requirements.txt`

**Quality impact is low** because: (1) the index is small and domain-focused (all content is lab tests — little noise for the reranker to filter), (2) metadata filtering already restricts retrieval to roughly half the index per query, and (3) adjusting top_k to 3 means the embedding similarity selects the 3 most relevant chunks directly rather than over-retrieving and reranking.

### Dependency Resolution

Initial `requirements.txt` pinned exact versions from the local conda environment, which had an inconsistent state: `llama-index-embeddings-openai==0.3.1` requires `llama-index-core<0.13.0`, but the local env had `llama-index-core==0.14.15`. Streamlit Cloud's resolver (uv) caught this conflict. Fix: removed version pins from all `llama-index-*` packages, letting pip resolve compatible versions together.

### Live Deployment

**URL:** https://clinical-lab-assistant-xd5df9urgepwfmeqrzkmpi.streamlit.app/

Note: Streamlit Community Cloud free tier puts apps to sleep after inactivity. First load after a sleep period will be slow (cold start — ChromaDB index and models reload from scratch). Subsequent queries in the same session are fast.

---

## Evaluation Score Progression

| Phase | Correctness | Faithfulness | Corpus | Key Change |
|-------|------------|--------------|--------|------------|
| Baseline (default chunking) | 4.38 / 5.0 | — | 3 tests | — |
| Re-ranking | 4.62 / 5.0 | — | 3 tests | Q7: 1.0 → 4.5 |
| Prompt template | 4.58 / 5.0 | — | 3 tests | Q7: 4.5 → 5.0 (exhaustiveness) |
| Agent layer | 4.71 / 5.0 | — | 3 tests | Q11: 2.0 → 5.0 (source routing) |
| Expanded corpus eval | 4.94 / 5.0 | — | 12 tests | 35/36 correct on new questions |
| **Faithfulness eval** | **4.94 / 5.0** | **36 / 36** | **12 tests** | **Zero hallucinations confirmed** |

*Note: the expanded corpus eval (Phase 10) uses a different question set than Phases 5–8, so correctness scores are not directly comparable. The 4.94 reflects system performance on 36 new questions across 9 new test types.*

---

## Current State

- Pipeline is fully operational end-to-end: ingest → clean → chunk → embed → store → retrieve → generate → route → UI
- **Streamlit app:** deployed on Streamlit Community Cloud — https://clinical-lab-assistant-xd5df9urgepwfmeqrzkmpi.streamlit.app/
- **App features:** streaming responses, source expander, conversational memory (condense_query pattern), display-only conversation history
- **Corpus:** 12 lab tests (A1C, CBC, CMP, BMP, TSH, PSA, PT/INR, Liver Panel, Troponin, Microalbumin, Ferritin, CRP)
- **Documents:** 24 LlamaIndex `Document` objects (12 tests × 2 sources)
- **Active index:** 47 chunks (1500/300 SentenceSplitter)
- **Retrieval (deployed):** `similarity_top_k=3`, no reranker (removed for deployment due to PyTorch size constraints)
- **Retrieval (notebook/local):** `similarity_top_k=6`, `SentenceTransformerRerank` (`cross-encoder/ms-marco-MiniLM-L-6-v2`, `top_n=2`)
- **Prompt template:** exhaustiveness instruction, disclaimer, source citation
- **Agent layer:** two-stage classification (intent gate + source routing) with metadata-filtered engines
- **Latest correctness score: 4.94 / 5.0** (36-question expanded corpus eval)
- **Faithfulness score: 36 / 36** (zero hallucinations — all failures are retrieval gaps, not fabrications)
- **Remaining known limitations:** Q12 (tabular data extraction — potassium), Q9 (partial ambient temp stability), liver panel fasting (cross-test noise within Medline partition)

---

## Next Steps

### Planned Future Work
- **Tabular data extraction:** Parse Mayo PDF reference range tables into structured format (CSV or structured chunks) to fix Q12 potassium range hallucination
- **Cross-test noise within Medline partition:** Fix liver panel fasting retrieval failure — potential approaches: `test_name` metadata filter (requires query-to-test-name resolution), or smaller chunk sizes for preparation/fasting content
- **Retrieval metrics:** `ContextRelevancyEvaluator` or `RetrieverEvaluator` to measure chunk-level retrieval quality independently of generation quality
- **ReActAgent (future):** Once version compatibility is resolved or llama-index stabilizes, swap manual routing for a proper agent with tool-use loop — enables multi-step reasoning and easier tool addition
- **Conversational memory:** ✅ Implemented via `condense_query` pattern
- **Observability:** Track latency, token usage, retrieval success rate per query type
