import argparse
import json
import re
from pathlib import Path
from typing import Iterator

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

DEFAULT_DOCS = Path("data/raw/fastapi/docs/en/docs")
DEFAULT_OUTPUT = Path("data/processed/fastapi_chunks.jsonl")
HEADERS_TO_SPLIT_ON = [("#", "h1"),
                       ("##", "h2"),
                       ("###", "h3"),
                       ("####", "h4")]
EXCLUDE_FILES = {"release-notes.md","_llm-test.md","external-links.md", "fastapi-people.md", "translation-banner.md"}  # trash
MIN_CHARS = 50  # remove small chunks

# Identificator Cleaner (from heads)
def clean_heading(text: str) -> str:
    return re.sub(r"\s*\{\s*#[^}]+\}\s*$", "", text).strip()


def build_chunks(doc_dir: Path, max_chars: int, overlap_chars: int) -> Iterator[dict]:
    # Splitters Setup
    # Head-level
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=True,
    )
    # Text-level
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=overlap_chars,
        strip_whitespace=False,
    )
    for path in sorted(doc_dir.rglob("*.md")):
        if path.name.startswith("_") or path.name in EXCLUDE_FILES:
            continue

        chunk_idx = 0
        default_title = path.stem.replace("-", " ").title()  # Default Title
        source = path.relative_to(doc_dir).as_posix()
        markdown = path.read_text(encoding="utf-8")

        sections = header_splitter.split_text(markdown)
        for chunk in text_splitter.split_documents(sections):
            text = chunk.page_content.strip()
            if len(text) < MIN_CHARS:
                continue
            chunk_idx += 1
            meta = chunk.metadata
            headers_path = [clean_heading(meta[head]) for _, head in HEADERS_TO_SPLIT_ON if meta.get(head)]  # get a Headers Path
            title = clean_heading(meta.get("h1", default_title))  # change title if possible

            # Context Injection (adding context (Headers Path) to the chunk text)
            if headers_path:
                text = f"{' > '.join(headers_path)}\n\n{text}"

            yield {
                "text": text,
                "metadata": {
                    "source": source,
                    "title": title,
                    "headers_path": headers_path,
                    "language": "en",
                    "chunk_index": chunk_idx,
                },
            }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS, help="FastAPI docs/en/docs directory")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL path")
    parser.add_argument("--max-chars", type=int, default=450, help="Maximum chunk size (characters)")
    parser.add_argument("--overlap-chars", type=int, default=50, help="Character overlap between chunks")
    args = parser.parse_args()

    if not args.docs.is_dir():
        parser.error(f"Docs directory not found: {args.docs}.")
    if args.max_chars < 1 or not 0 <= args.overlap_chars < args.max_chars:
        parser.error("Require max-chars > 0 and 0 <= overlap-chars < max-chars.")

    if not args.output.parent.exists():
        args.output.parent.mkdir(parents=True)

    count = 0
    with args.output.open("w", encoding="utf-8") as output:
        for chunk in build_chunks(args.docs, args.max_chars, args.overlap_chars):
            output.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            count += 1
    print(f"Wrote {count} chunks to {args.output}")


if __name__ == "__main__":
    main()