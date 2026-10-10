# FastAPI IT Assistant (RAG)

A small local RAG assistant that answers questions using the official FastAPI documentation. It creates Markdown chunks, retrieves relevant chunks with embeddings, and asks a local language model to write an answer from that context.

## Requirements

- Python 3.9 or newer
- [Ollama](https://ollama.com/download) for local answer generation
- The FastAPI documentation repository

Create a virtual environment and install the Python packages:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## 1. Get the FastAPI documentation

From the project root, clone the repository:

```bash
git clone --depth 1 https://github.com/fastapi/fastapi.git data/raw/fastapi
```

The Markdown files used by the project are in `data/raw/fastapi/docs/en/docs`.

## 2. Create chunks

Run the chunker from the project root:

```bash
.venv/bin/python src/chunking.py
```

This writes `data/processed/fastapi_chunks.jsonl`. Each line contains a chunk's text and metadata: source file, title, heading path, language, and chunk number within that file.

The defaults are a maximum of 1,200 characters per chunk, a 150-character overlap, and a minimum chunk length of 50 characters (you can also change the chunk size and overlap):

```bash
.venv/bin/python src/chunking.py \
  --max-chars 1200 \
  --overlap-chars 150
```

## 3. Build embeddings and search

Build the local embedding file:

```bash
.venv/bin/python src/retrieval.py build
```

This uses `intfloat/multilingual-e5-small` and writes `data/processed/fastapi_embeddings.npy`. The first run downloads the embedding model. If you change the chunks, rebuild the embeddings before searching.

Search for relevant chunks without generating an answer:

```bash
.venv/bin/python src/retrieval.py search "How do I add a response model?"
```

The search command prints the similarity score, source path, heading path, and chunk text for the top five results.

## 4. Run the local RAG assistant

Install and start Ollama system, then download the default local language model:

```bash
ollama pull qwen2.5:1.5b
```

Ask a question from the project root:

```bash
.venv/bin/python src/it_assistant/rag.py \
  "How do I define an async endpoint that awaits an async library?"
```

The assistant retrieves the top five chunks, sends them with the question to Ollama running locally, and prints the generated answer and source file paths. No paid model API is used.

You can change how many chunks are passed to the language model:

```bash
.venv/bin/python src/it_assistant/rag.py \
  "How do I add a response model?" \
  --top-k 3
```

## 5. Evaluation questions

**Example 1:**

Question: "How do I add a dependency to a path operation if I don't need its return value?"

Answer:
If you don't need the return value of a dependency inside your path operation function, but still need it to be executed, you can add a `list` of `dependencies` to the path operation decorator instead of declaring a path operation function parameter with `Depends`.

Sources:
- tutorial/dependencies/dependencies-in-path-operation-decorators.md
- tutorial/dependencies/index.md

**Example 2:**

Question: "How do I define an async endpoint that awaits an async library?" 

Answer:
You can define an async endpoint that awaits an async library using the `async def` keyword. For example:

```Python
@app.get('/')
async def read_results():
    results = await some_async_library()
    return results
```

You would replace `some_async_library` with the name of the library you are awaiting. Make sure you declare the library function as `async def` to indicate that it is asynchronous.

Sources:
- async.md
- tutorial/dependencies/index.md

**Example 3:**

Question: "How do I define path parameters and query parameters in a FastAPI route?"

Answer:
You can declare multiple path parameters and query parameters at the same time, **FastAPI** knows which is which. This is done by assigning them to the parameters list in the route decorator. For example:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}
```

In this example, `item_id` is a path parameter, while `Query` is used for query parameters. The parameter is automatically recognized and mapped to the appropriate part of the request.

Sources:
- tutorial/query-params.md
- tutorial/body.md
- tutorial/query-param-models.md
- tutorial/first-steps.md
- reference/parameters.md

**Example 4:**

Question: "How do I create a Flutter mobile app UI that directly renders FastAPI components?"

Answer:
You could not find information.

Sources:
- tutorial/first-steps.md
- fastapi-cli.md
- advanced/dataclasses.md
- contributing.md

**Example 5:**

Question: "How do I use Blade templating engine with FastAPI routes?"

Answer:
Blade templating engine cannot be used with FastAPI.

Sources:
- advanced/templates.md
- editor-support.md
- tutorial/first-steps.md
- reference/templating.md