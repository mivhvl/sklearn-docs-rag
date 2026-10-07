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

    model = SentenceTransformer(model_name) # Autoencoder (automatically calculate entire sentence embeddings)
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


def search(chunks_path: Path, embeds_path: Path, query: str, top_k: int, model_name: str) -> None:
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
    best_indices = np.argsort(-scores)[:top_k]

    for index in best_indices:
        chunk = chunks[int(index)] # get embedding
        metadata = chunk["metadata"] # get metadata
        print(f"\nScore={scores[index]:.3f} | {metadata.get('source', 'unknown')}") 
        print(chunk["text"])


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
