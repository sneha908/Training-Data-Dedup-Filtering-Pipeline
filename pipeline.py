import os
import json
import hashlib
import subprocess
import time
from datetime import datetime, timezone

import torch
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from datasketch import MinHash, MinHashLSH
import faiss
from sentence_transformers import SentenceTransformer
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from transformers import pipeline
from tqdm import tqdm

print("All imports successful")
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

def get_git_commit():
    """Best-effort git commit hash for provenance. Returns 'unknown' outside a repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def shingles(text, k=5):
    """Word k-gram shingles, not a unique-word set. Preserves local word order,
    which is what MinHash/Jaccard similarity is meant to operate over."""
    words = text.lower().split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


class DataDedupPipeline:
    def __init__(
        self,
        output_dir="data/shards",
        checkpoint_file="checkpoints/state.json",
        manifest_dir="manifests",
        min_words=5,
        near_dup_threshold=0.8,
        semantic_threshold=0.92,
        num_perm=128,
        shingle_size=5,
        use_quality_classifier=False,
        quality_classifier_min_score=0.9,
    ):
        self.output_dir = output_dir
        self.checkpoint_file = checkpoint_file
        self.manifest_dir = manifest_dir
        self.min_words = min_words
        self.near_dup_threshold = near_dup_threshold
        self.semantic_threshold = semantic_threshold
        self.num_perm = num_perm
        self.shingle_size = shingle_size
        self.use_quality_classifier = use_quality_classifier
        self.quality_classifier_min_score = quality_classifier_min_score

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.checkpoint_file) or ".", exist_ok=True)
        os.makedirs(self.manifest_dir, exist_ok=True)

        # 1. Exact-dup hash set
        self.seen_hashes = set()

        # 2. Near-dup MinHash LSH (shingled)
        self.lsh = MinHashLSH(threshold=self.near_dup_threshold, num_perm=self.num_perm)

        # 3. Semantic dedup: FAISS + SentenceTransformer, IndexFlatIP for cosine
        #    similarity on normalized embeddings. Flat (exact) is fine at this
        #    scale; would move to IVF/HNSW for tens of millions+ of vectors.
        print("Loading SentenceTransformer model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
        
        # Base CPU Index
        cpu_index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Move FAISS index to GPU if available
        if device == "cuda":
            print("Moving FAISS index to GPU...")
            self.res = faiss.StandardGpuResources()  # Temporary RAM resource for GPU operations
            self.faiss_index = faiss.index_cpu_to_gpu(self.res, 0, cpu_index)
        else:
            print("CUDA not available. Falling back to CPU FAISS index.")
            self.faiss_index = cpu_index

        # 4. PII stripping (Note: Presidio runs strictly on CPU)
        print("Loading PII analyzer...")
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self.pii_entities = ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "LOCATION", "CREDIT_CARD"]

        # 5. Quality classifier (real, wired in — used only if enabled)
        self.quality_classifier = None
        if self.use_quality_classifier:
            print("Loading quality classifier...")
            self.quality_classifier = pipeline(
                "text-classification",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                truncation=True,
                device=0 if device == "cuda" else -1,
            )

        self.stats = {
            "total_processed": 0,
            "quality_dropped": 0,
            "exact_dup_dropped": 0,
            "near_dup_dropped": 0,
            "semantic_dup_dropped": 0,
            "saved": 0,
        }
        self.run_started_at = datetime.now(timezone.utc).isoformat()

    # ---------- checkpointing ----------

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, "r") as f:
                return json.load(f).get("last_processed_idx", 0)
        return 0

    def save_checkpoint(self, idx):
        with open(self.checkpoint_file, "w") as f:
            json.dump({"last_processed_idx": idx}, f)

    # ---------- core stages ----------

    def passes_quality_filter(self, text):
        if len(text.split()) < self.min_words:
            return False
        if self.quality_classifier is not None:
            score = self.quality_classifier(text[:512])[0]
            if score["label"] == "NEGATIVE" and score["score"] > self.quality_classifier_min_score:
                return False
        return True

    def is_exact_duplicate(self, text):
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if h in self.seen_hashes:
            return True, h
        self.seen_hashes.add(h)
        return False, h

    def is_near_duplicate(self, text, key):
        m = MinHash(num_perm=self.num_perm)
        for shingle in shingles(text, self.shingle_size):
            m.update(shingle.encode("utf-8"))
        if self.lsh.query(m):
            return True
        self.lsh.insert(key, m)
        return False

    def process_chunk(self, chunk, chunk_id):
        """Runs quality -> exact -> near-dup -> semantic (sequential, one at a
        time so each new record is checked against records already accepted
        in this same batch, not just prior batches) -> PII."""
        
        # Phase 1: Heavy Filtering on CPU (Fast text/hash operations)
        pre_filtered_records = []
        for record in chunk:
            self.stats["total_processed"] += 1
            text = record.get("text", "")

            if not self.passes_quality_filter(text):
                self.stats["quality_dropped"] += 1
                continue

            is_dup, exact_hash = self.is_exact_duplicate(text)
            if is_dup:
                self.stats["exact_dup_dropped"] += 1
                continue

            if self.is_near_duplicate(text, key=exact_hash):
                self.stats["near_dup_dropped"] += 1
                continue
                
            pre_filtered_records.append(record)

        if not pre_filtered_records:
            return

        # Phase 2: Batch Embeddings on GPU (Massive Speedup over single-sentence calls)
        texts_to_embed = [r.get("text", "") for r in pre_filtered_records]
        embeddings = self.embedder.encode(texts_to_embed, batch_size=64, show_progress_bar=False, normalize_embeddings=True)

        # Phase 3: Semantic Dedup & Final PII Scrub
        valid_records = []
        
        for idx, record in enumerate(pre_filtered_records):
            text = record.get("text", "")
            embedding = embeddings[idx : idx + 1]  # Extract single vector as a 2D array for FAISS

            # Query GPU FAISS Index
            if self.faiss_index.ntotal > 0:
                distances, _ = self.faiss_index.search(embedding, 1)
                if distances[0][0] > self.semantic_threshold:
                    self.stats["semantic_dup_dropped"] += 1
                    continue
            
            # Immediately add to GPU index so the next loop iterations can find it
            self.faiss_index.add(embedding)

            # PII stripping (Note: This engine executes on CPU)
            results = self.analyzer.analyze(text=text, entities=self.pii_entities, language="en")
            anonymized_text = self.anonymizer.anonymize(text=text, analyzer_results=results).text

            valid_records.append({"original_id": record.get("id", ""), "text": anonymized_text})

        if valid_records:
            self.stats["saved"] += len(valid_records)
            table = pa.Table.from_pylist(valid_records)
            shard_path = os.path.join(self.output_dir, f"shard_{chunk_id:05d}.parquet")
            pq.write_table(table, shard_path)

    def write_manifest(self, dataset_name=None, subset=None, split=None, chunk_size=None, elapsed_seconds=None):
        """Writes a final execution receipt capturing git state and drop metrics."""
        manifest = {
            "run_started_at": self.run_started_at,
            "run_ended_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed_seconds, 1) if elapsed_seconds else None,
            "git_commit": get_git_commit(),
            "source_dataset": {"name": dataset_name, "subset": subset, "split": split},
            "config": {"chunk_size": chunk_size},
            "stats": self.stats,
        }
        manifest_path = os.path.join(self.manifest_dir, f"manifest_{int(time.time())}.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Manifest written to {manifest_path}")
        print("Final Pipeline Stats:", json.dumps(self.stats, indent=2))

    def run(self, dataset_name, subset, split="train", chunk_size=1000, max_records=None):
        print(f"Loading streaming dataset: {dataset_name}...")
        dataset = load_dataset(dataset_name, subset, split=split, streaming=True)

        start_idx = self.load_checkpoint()
        chunk = []
        chunk_id = start_idx // chunk_size
        idx = start_idx - 1
        started = time.time()

        iterator = iter(dataset)
        if start_idx > 0:
            print(f"Fast-forwarding to record {start_idx}...")
            for _ in tqdm(range(start_idx), desc="Skipping"):
                next(iterator, None)

        print("Starting processing pipeline...")
        try:
            for idx, record in enumerate(tqdm(iterator, initial=start_idx), start=start_idx):
                if "text" not in record:
                    continue
                chunk.append(record)

                if max_records and idx + 1 >= max_records:
                    self.process_chunk(chunk, chunk_id)
                    self.save_checkpoint(idx + 1)
                    print(f"\nReached max_records={max_records}, stopping.")
                    break

                if len(chunk) >= chunk_size:
                    self.process_chunk(chunk, chunk_id)
                    self.save_checkpoint(idx + 1)
                    chunk = []
                    chunk_id += 1
                    if chunk_id % 10 == 0:
                        print(f"\nStats: {self.stats}")

            if chunk and not (max_records and idx + 1 >= max_records):
                self.process_chunk(chunk, chunk_id)
                self.save_checkpoint(idx + 1)

        except KeyboardInterrupt:
            print("\nPipeline interrupted by user. State saved for resume.")

        elapsed = time.time() - started
        print("\n=== Final Stats ===")
        print(json.dumps(self.stats, indent=2))
        reduction = 100 - ((self.stats["saved"] / max(1, self.stats["total_processed"])) * 100)
        print(f"Corpus reduced by {reduction:.2f}%")

        self.write_manifest(dataset_name, subset, split, chunk_size, elapsed)

  if __name__ == '__main__':
    pipeline_app = DataDedupPipeline(
        use_quality_classifier=False  # flip on once you've validated it against your data
    )
    pipeline_app.run(dataset_name="allenai/c4", subset="en", chunk_size=500, max_records=150_000)
