import json
from urllib.request import Request, urlopen

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_LLM_MODEL = "qwen2.5:1.5b"


def build_prompt(question: str, results: list[dict]) -> str:
    context_parts = []

    for result in results:
        source = result["metadata"].get("source", "unknown")
        context_parts.append(f"Source: {source}\n{result['text']}")

    context = "\n---\n".join(context_parts)

    return f"""Answer the question using only the context provided below.
If the context does not contain the right answer, say that you could not find information.
Also mention the source file in your answer.

Context:
{context}

Question: {question}

Answer:"""

def generate_answer(prompt: str, model_name: str = DEFAULT_LLM_MODEL) -> str:
    request_data = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
    }
    request = Request(
        OLLAMA_URL,
        data=json.dumps(request_data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    with urlopen(request, timeout=180) as response:
        result = json.load(response)

    return result["response"].strip()
