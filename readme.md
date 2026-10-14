
Claude finished the response
Ran 5 commands, edited a file, and 2 more tools
Removed Harry (unrelated to this role anyway — general RAG, not training-data work), added the dedup pipeline project with your real numbers, merged the skills additions into the existing ML line instead of a new one to keep it at one page. Everything else — experience, TRACKR, PR review system, education, achievements — untouched.

One thing to flag before you submit: the GitHub link in the header for this project won't work until you actually push the repo per the steps from earlier — do that before this resume goes anywhere.


Resume
Document·PDF 

Resume
TEX 
provide a LINK option or something by which the recruiter can visit the project

Adding project links to the resume like existing entries.
Adding project links to the resume like existing entries.
I can add a clickable GitHub link next to the project title (same pattern used for TRACKR's live link) — but I need the actual repo URL, since I can't invent one you haven't created yet.

What's the repo URL going to be? If you haven't pushed it yet but know the name, give me github.com/AbhinavGangwar12/<repo-name> and I'll wire it in now — just make sure the repo is actually live and public before this resume goes out, or a recruiter clicking it gets a 404.

for trackr the url is https://trackr-ivory-eight.vercel.app/fitness-tracker
for pr review it is gonna be https://github.com/AbhinavGangwar12/pr-review-agent-tom
and for the pipeline, it will be a kaggle notebook link https://www.kaggle.com/code/abnav1203/pipeline

Edited a file, ran 2 commands, shared files
All three project titles are now clickable (underlined, same style as your header links). One flag: a Kaggle code/notebook link for the pipeline project is weaker than a GitHub repo — recruiters expect to see clean, reviewable source + README on GitHub, not scroll through a notebook. If you have time before submitting, push the actual GitHub repo (as planned earlier) and swap that link in — it's a stronger signal for a data-engineering role specifically. Also double check the PR-review repo name (pr-review-agent-tom) and the Kaggle notebook are both public before this goes out, or the links 404 for the recruiter.


Resume
Document·PDF 

Resume
TEX 
ok i will create a github repo for the pipeline as well, provide a readme for it

Edited a file, read a file, and 2 more tools
Updated to match your actual GPU pipeline and real run: batched GPU embeddings, max_records cap explained, and a "Results" section with your real 150K-record stats and the reasoning behind them (pre-deduplicated source, semantic dedup catching what hash-based methods miss).

When you push the repo: include pipeline.py, README.md, requirements.txt, your real manifest.json from that run, and ideally 1–2 sample output shards. Once it's live, swap the Kaggle link on the resume for the GitHub repo URL — that's the stronger link for this role.


Readme
Document·MD 

Write a message…


Claude is AI and can make mistakes.
SELECT

EXPORT



Readme · MD
# Training Data Dedup & Filtering Pipeline
 

A multi-stage pipeline for cleaning large text corpora before LLM training:
quality filtering, exact/near-duplicate/semantic deduplication, PII
stripping, versioned Parquet output, and a per-run provenance manifest.
 

## Pipeline stages
 

1. **Quality filtering** — length heuristic; optional classifier-based
   scoring (toggle via `use_quality_classifier`).
2. **Exact deduplication** — SHA-256 hash set.
@@ -78,8 +22,9 @@ stripping, versioned Parquet output, and a per-run provenance manifest.
   per run recording the git commit, all filter thresholds, the embedding
   model version, and final stats — so "what went into this data" is
   answerable from one file, not a memory of what the script did that day.

## Design notes
 

- **Streaming, not in-memory.** Uses HuggingFace `datasets` in
  `streaming=True` mode so the source corpus never has to fit in RAM.
- **Resumable.** A JSON checkpoint tracks the last processed record index;
@@ -90,11 +35,12 @@ stripping, versioned Parquet output, and a per-run provenance manifest.
  source (C4's `en` split alone is ~365M records) in a single session.
- **Order of operations matters.** PII stripping runs *after* dedup, so we
  never spend PII-scrubbing cost on text about to be discarded.

## Results (sample run)
 

Run on 150,000 streamed records from `allenai/c4` (`en` split), GPU-accelerated
on a Kaggle T4:
 

```
total_processed:      150,000
quality_dropped:            0
@@ -104,31 +50,31 @@ semantic_dup_dropped:     172
saved:                 149,429
corpus reduction:        0.38%
```
 

`exact_dup_dropped` and `quality_dropped` are near-zero because C4 is already
deduplicated and length-filtered upstream by its authors — a fresh hash pass
over it finds little. The 172 `semantic_dup_dropped` records are the
interesting number: these are near-identical / reworded records that hash-based
methods (exact hash, MinHash) miss entirely, caught only because of the
embedding-based dedup stage. On a dirtier, non-pre-deduplicated corpus (raw
CommonCrawl shards, for example), reduction rates are typically far higher.
 

Full run manifest (git commit, thresholds, model versions, timing) is in
`manifests/`.
 

## Usage
 

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg  # required by Presidio
python pipeline.py
```
 

Or import and configure directly:
 

```python
from pipeline import DataDedupPipeline
 

pipeline = DataDedupPipeline(
    output_dir="data/shards",
    near_dup_threshold=0.8,
@@ -142,14 +88,14 @@ pipeline.run(
    max_records=150_000,  # omit to run unbounded (not recommended for C4)
)
```
 

Runs on GPU automatically if CUDA is available (embeddings + FAISS index +
optional quality classifier); falls back to CPU otherwise. See
[the Kaggle notebook](https://www.kaggle.com/code/abnav1203/pipeline) for a
runnable, GPU-enabled example.
 

## Known limitations (deliberate scope choices, not oversights)
 

- Semantic dedup uses an exact FAISS `IndexFlatIP`, which is fine at this
  scale but would need to move to IVF/HNSW for tens of millions+ of
  vectors.
@@ -158,13 +104,3 @@ runnable, GPU-enabled example.
- Quality classifier is a placeholder (SST-2 sentiment model) standing in
  for a real quality/toxicity classifier — swap in a fastText or
  perplexity-based quality model before using this on real training data.
 
