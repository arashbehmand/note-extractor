#!/usr/bin/env python3
"""
Document Summarizer

This script summarizes input documents using LangChain and OpenAI's GPT models.
It processes the document, splits it into chunks, and generates a summary using
a map-reduce approach.

Usage:
    python document_summarizer.py [-h] [-i INPUT] [-o OUTPUT] [-c CACHE_DIR] [-k API_KEY]

For more information, see the README.md file.
"""

import argparse
import hashlib
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_unstructured import UnstructuredLoader
from tqdm import tqdm

# Define the map prompt template
MAP_PROMPT_TEMPLATE = """
You are an expert at summarizing complex documents into concise, handwritten-style notes that capture the big picture and key points.

Instructions:
- Output the notes in **Markdown** format.
- Include **mermaid diagrams** in mindmap type or any other useful type, when they help to explain concepts from the original text. [beware of the screen width with diagrams]
- Include **formulas** using LaTeX syntax only if present in the original text and only the key formulas not all formulas.
- Structure the notes into multiple levels of depth, similar to a mind map:
  - **First depth**: Provide an overview in a few sentences.
  - **Second depth**: Explain main topics with single sentences.
  - **Third depth**: List detailed points as keywords.
  - ...
- Use bullet points, headings, and subheadings to organize content.
- Avoid full sentences unless necessary; prefer phrases and keywords.
- Emphasize important concepts and relationships.

Given the following text:
{text}

So far, the summary is:
{summary}

Produce the notes in this style:
"""

# Define the TL;DR summary prompt template
TLDR_PROMPT_TEMPLATE = """
You are an expert at summarizing complex documents into concise summaries. You will receive a summary of previous chunks and the text of the current chunk. Your task is to produce an updated summary that incorporates both the previous summary and the new information.

Instructions:
- Provide a single-paragraph TL;DR summary that combines the essence of the previous summary with the main points from the new text.
- Ensure the new summary maintains overall context and continuity.
- The output should be a cohesive summary that replaces the previous one entirely.

Previous TL;DR summary:
{summary}

New text to incorporate:
{text}

Provide the updated TL;DR summary below:
"""

# Define the reduce prompt template
REDUCE_PROMPT_TEMPLATE = """
You will receive several chunks of notes generated from a larger document. Your task is to combine these notes into a coherent, concise, and consistent set of study notes.

Instructions:
- Ensure the combined notes are consistent and free of duplicate information.
- Maintain the Markdown format.
- Keep the structure with three levels of depth as previously described.
- Ensure that diagrams and formulas are correctly placed and do not duplicate content.
- Make sure the final notes are usable as a study guide.

Given the following notes:

{notes}

Combine and refine these notes into a single, cohesive set of notes.
"""

CONSISTENCY_PROMPT_TEMPLATE = """
Review the following notes for consistency, coherence, and to ensure there is no duplicate information. Make any necessary adjustments to improve clarity and usefulness as a study guide.

Notes:

{final_notes}

Provide the refined notes as a Markdown file, adhering to these strict rules:
1. Start directly with the highest-level heading (e.g., # Chapter Title).
2. Do not include any introductory or concluding statements.
3. Do not enclose the content of summary in code blocks or use any surrounding backticks.
4. Ensure all headings use the appropriate Markdown syntax (# for main headings, ## for subheadings, etc.).
5. Use proper Markdown formatting for lists, emphasis, and code snippets if needed.
6. End the content with the last relevant point or heading, without any closing remarks.

Your response should be a ready-to-use Markdown file:
"""

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_env_var(var_name, default=None):
    """Retrieve environment variable or return default value."""
    return os.getenv(var_name, default)


def clean_filename(filename):
    """Clean up filename for use in cache directory naming."""
    return "".join(c if c.isalnum() else "_" for c in filename)


def get_file_hash(file_path):
    """Generate MD5 hash of file content."""
    with open(file_path, "rb") as f:
        file_hash = hashlib.md5()
        chunk = f.read(8192)
        while chunk:
            file_hash.update(chunk)
            chunk = f.read(8192)
    return file_hash.hexdigest()[:8]


def load_document(file_path):
    """Load document using UnstructuredLoader."""
    api_url = get_env_var(
        "UNSTRUCTURED_API_URL", "https://api.unstructured.io/general/v0/general"
    )
    api_key = get_env_var("UNSTRUCTURED_API_KEY")
    use_api = get_env_var("USE_UNSTRUCTURED_API", "false").lower() == "true"

    loader = UnstructuredLoader(
        file_path,
        url=api_url,
        api_key=api_key,
        partition_via_api=use_api,
        chunking_strategy="by_title",
        strategy="fast",
    )
    logger.info("Loading document...")
    return loader.load()


def split_text(document):
    """Split document into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=15000, chunk_overlap=1000)
    logger.info("Splitting text into chunks...")
    return text_splitter.split_documents([document])


def process_chunks(texts, cache_dir, map_chain, tldr_chain):
    """Process text chunks and collect notes."""
    mapped_notes = []
    summary_so_far = [""]
    for i, chunk in enumerate(tqdm(texts)):
        chunk_filename = cache_dir / f"chunk_{i}.txt"
        mapped_notes_filename = cache_dir / f"mapped_notes_{i}.txt"
        tldr_filename = cache_dir / f"tldr_{i}.txt"

        if not chunk_filename.exists():
            chunk_filename.write_text(chunk.page_content, encoding="utf-8")

        if not mapped_notes_filename.exists():
            response = map_chain.invoke(
                {"text": chunk.page_content, "summary": summary_so_far[-1]}
            )
            mapped_note = response.content
            mapped_notes_filename.write_text(mapped_note, encoding="utf-8")
        else:
            mapped_note = mapped_notes_filename.read_text(encoding="utf-8")
        mapped_notes.append(mapped_note)

        if not tldr_filename.exists():
            tldr_response = tldr_chain.invoke(
                {"text": chunk.page_content, "summary": summary_so_far[-1]}
            )
            tldr_summary = tldr_response.content
            tldr_filename.write_text(tldr_summary, encoding="utf-8")
        else:
            tldr_summary = tldr_filename.read_text(encoding="utf-8")
        summary_so_far.append(tldr_summary)

    return mapped_notes, summary_so_far


def main(input_args):
    """Main function to run the document summarizer."""
    input_file = Path(input_args.input)
    output_file = (
        Path(input_args.output)
        if input_args.output
        else input_file.with_name(f"{input_file.stem}_summary.md")
    )
    cache_dir = (
        Path(input_args.cache_dir)
        if input_args.cache_dir
        else Path(
            f"_cache_{clean_filename(input_file.name)}_{get_file_hash(input_file)}"
        )
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    openai_api_key = input_args.api_key or get_env_var("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError(
            "OpenAI API key is required. Provide it via -k argument or set OPENAI_API_KEY environment variable."
        )

    # Load and process document
    documents = load_document(input_file)
    combined_content = "\n".join([doc.page_content for doc in documents])
    combined_document = Document(page_content=combined_content)
    texts = split_text(combined_document)

    # Initialize LLMs and chains
    llm_mini = ChatOpenAI(
        model_name="gpt-4o-mini", temperature=0, api_key=openai_api_key
    )
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.1, api_key=openai_api_key)

    map_prompt = PromptTemplate(
        template=MAP_PROMPT_TEMPLATE, input_variables=["text", "summary"]
    )
    map_chain = map_prompt | llm_mini

    tldr_prompt = PromptTemplate(
        template=TLDR_PROMPT_TEMPLATE, input_variables=["text", "summary"]
    )
    tldr_chain = tldr_prompt | llm_mini

    # Process chunks
    mapped_notes, _ = process_chunks(texts, cache_dir, map_chain, tldr_chain)

    # Combine notes
    reduce_prompt = PromptTemplate(
        template=REDUCE_PROMPT_TEMPLATE, input_variables=["notes"]
    )
    reduce_chain = reduce_prompt | llm

    notes_str = "\n\n".join(mapped_notes)
    final_notes = reduce_chain.invoke({"notes": notes_str})

    # Consistency check
    consistency_prompt = PromptTemplate(
        template=CONSISTENCY_PROMPT_TEMPLATE, input_variables=["final_notes"]
    )
    consistency_chain = consistency_prompt | llm

    refined_notes = consistency_chain.invoke({"final_notes": final_notes.content})

    # Write output
    output_file.write_text(refined_notes.content, encoding="utf-8")
    logger.info("Summary written to %s", output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarize a document using LangChain and OpenAI's GPT models."
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Path to the input document"
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to the output summary file (default: {input_file}_summary.md)",
    )
    parser.add_argument("-c", "--cache_dir", help="Directory to store cache files")
    parser.add_argument("-k", "--api_key", help="OpenAI API key")
    args = parser.parse_args()

    main(args)
