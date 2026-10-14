

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
 
