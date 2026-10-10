import argparse
import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_CHUNKS = Path("data/processed/fastapi_chunks.jsonl")
DEFAULT_EMBEDS = Path("data/processed/fastapi_embeddings.npy")

def load_chunks(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]

def build_embeds(chunks_path: Path, embeds_path: Path, model_name: str) -> None:
    if not chunks_path.is_file():
        raise SystemExit(f"Chunks file not found.")

    chunks = load_chunks(chunks_path)
    if not chunks:
        raise SystemExit(f"No chunks found.")

    model = SentenceTransformer(model_name) # Autoencoder (automatically calculates entire sentence embeddings)
    # Sentence Embeddings 
    embeddings = model.encode(
        [f"passage: {chunk['text']}" for chunk in chunks], # add 'passage:' before the chunk
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    # Save Embeddings
    embeds_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeds_path, embeddings)
    print(f"Embedded {len(chunks)} chunks saved in {embeds_path}")


def retrieve(chunks_path: Path, embeds_path: Path, query: str, top_k: int, model_name: str) -> list[dict]:
    if not chunks_path.is_file():
        raise SystemExit(f"Chunks file not found.")
    if not embeds_path.is_file():
        raise SystemExit(f"Embeddings not found. Build it first with the 'build' command.")
    if top_k < 1:
        raise SystemExit("--top-k must be >= 1")

    chunks = load_chunks(chunks_path)
    embeddings = np.load(embeds_path)

    model = SentenceTransformer(model_name)
    query_embedding = model.encode(
        f"query: {query}",                  # add 'query:' before the question
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    # Scoring the question with chunk's embeddings and taking k indexed with the best score
    scores = embeddings @ query_embedding
    best_chunks_idx = np.argsort(-scores)[:top_k]

    results = []
    for index in best_chunks_idx:
        chunk = chunks[int(index)] # get embedding
        results.append({
            "text": chunk["text"],
            "metadata": chunk["metadata"],
            "score": float(scores[index]),
        })

    return results

def search(chunks_path: Path, index_path: Path, query: str, top_k: int, model_name: str) -> None:
    
    results = retrieve(chunks_path, index_path, query, top_k, model_name)

    for result in results:
        metadata = result["metadata"]
        heading_path = " > ".join(metadata.get("headers_path", []))
        print(f"\n\nScore={result['score']:.2f} | {metadata.get('source', 'unknown')}")
        if heading_path:
            print(heading_path)
        print(result["text"])


def main() -> None:
    # Global args
    parser = argparse.ArgumentParser(description='Arguments for a Retriever Setup/Usage')
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS, help="Input chunks JSONL")
    parser.add_argument("--embeds", type=Path, default=DEFAULT_EMBEDS, help="Local embeddings path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Sentence Transformers model name")
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    # Build Parser
    build_parser = subparsers.add_parser("build", help="Create embeddings for all chunks")
    # Search parser
    search_parser = subparsers.add_parser("search", help="Find the most relevant chunks")
    search_parser.add_argument("query", help="Question or search text")
    search_parser.add_argument("--top-k", type=int, default=5, help="Number of results to show")

    args = parser.parse_args()

    if args.command == "build":
        build_embeds(args.chunks, args.embeds, args.model) # embedding
    elif args.command == "search":
        search(args.chunks, args.embeds, args.query, args.top_k, args.model) #retriever


if __name__ == "__main__":
    main()
