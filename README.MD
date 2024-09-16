# Document Summarizer

This CLI application summarizes input documents using LangChain and OpenAI's GPT models. It processes the document, splits it into chunks, and generates a summary using a map-reduce approach.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/document-summarizer.git
   cd document-summarizer
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the project root and add the following variables:
   ```
   OPENAI_API_KEY=your_openai_api_key
   UNSTRUCTURED_API_URL=https://api.unstructured.io/general/v0/general
   UNSTRUCTURED_API_KEY=your_unstructured_api_key
   USE_UNSTRUCTURED_API=false
   ```

## Usage

Run the script with the following command:

```
python document_summarizer.py [-h] -i INPUT [-o OUTPUT] [-c CACHE_DIR] [-k API_KEY]
```

Arguments:
- `-i`, `--input`: Path to the input document (required)
- `-o`, `--output`: Path to the output summary file (default: {input_file}_summary.md)
- `-c`, `--cache_dir`: Directory to store cache files
- `-k`, `--api_key`: OpenAI API key (can also be set via OPENAI_API_KEY environment variable)

Example:
```
python document_summarizer.py -i path/to/your/document.pdf -o path/to/output/summary.md
```

## Features

- Processes various document formats using Unstructured
- Splits large documents into manageable chunks
- Uses OpenAI's GPT models for summarization
- Implements a map-reduce approach for efficient processing
- Caches intermediate results for faster reprocessing
- Generates a final summary in Markdown format

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `UNSTRUCTURED_API_URL`: URL for the Unstructured API
- `UNSTRUCTURED_API_KEY`: Your Unstructured API key
- `USE_UNSTRUCTURED_API`: Set to "true" to use the Unstructured API, "false" to process locally

## License

This project is licensed under the LGPL License - see the [LICENSE](LICENSE) file for details.