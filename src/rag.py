
import argparse
from pathlib import Path
from generation import DEFAULT_LLM_MODEL, build_prompt, generate_answer
from retrieval import DEFAULT_CHUNKS, DEFAULT_EMBEDS, DEFAULT_MODEL, retrieve


def main() -> None:
    parser = argparse.ArgumentParser(description='RAG Model')
    parser.add_argument("question", help="Question to ask about FastAPI")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS, help="Input chunks JSONL")
    parser.add_argument("--embeds", type=Path, default=DEFAULT_EMBEDS, help="Local embeddings file")
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL, help="Model used for chunks")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL, help="Local Ollama model for answers")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to create a promt")
    args = parser.parse_args()

    results = retrieve(args.chunks, args.embeds, args.question, args.top_k, args.embedding_model)
    prompt = build_prompt(args.question, results)
    answer = generate_answer(prompt, args.llm_model)

    print('\n\nAnswer:')
    print(answer)
    print("\nSources:")
    sources = dict.fromkeys(result["metadata"].get("source", "unknown") for result in results)
    for source in sources:
        print(f"- {source}")


if __name__ == "__main__":
    main()
