# File-Summarizer: Summarize File Contents with LLM

This project provides a Python-based solution to summarize meeting transcripts using a Large Language Model (LLM). The script processes a text file containing a transcript and generates a concise summary, including key points, speaker actions, follow-ups, and other essential details.

## Features

- **Token Counting**: Efficiently counts the number of tokens in the transcript using `tiktoken`.
- **Customizable Summarization**: Allows custom prompts for generating summaries with bullet points.
- **LLM Integration**: Utilizes the `OllamaLLM` model (Mistral) for text summarization with adjustable context size and predictions.
- **Easy-to-Use CLI**: Accepts a text file as input via the command line and outputs the summary directly.

## Prerequisites

Before running the script, ensure you have the following:

- Python 3.7+
- Required Python libraries:
  - `argparse`
  - `langchain`
  - `tiktoken`
  - `idlelib`
- Access to the `OllamaLLM` or another compatible LLM model.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/lamb-russell/file-summarizer
   cd file-summarizer
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Prepare a text file containing your meeting transcript.
2. Run the script with the file path as an argument:
   ```bash
   python main.py /path/to/your/transcript.txt
   ```
3. The script will output:
   - Token count of the input text.
   - A summary of key points.

### Example

Input:

A meeting transcript file:

```
Tom from product and Jerry from engineering met to discuss project "Mallet". Tom insisted Jerry complete the project next Tuesday. Jerry proposed Thursday instead. Jerry promised to follow up with Tom about project Mallet next Tuesday.
```

Command:

```bash
python summarize_transcripts.py meeting_transcript.txt
```

Output:

```
Token count of the transcript: 42

=== SUMMARY OF KEY POINTS ===
1. Tom from product and Jerry from engineering met to discuss project "Mallet".
2. Tom insisted Jerry complete the project next Tuesday. Jerry proposed Thursday instead.
3. Jerry promised to follow up with Tom about project Mallet next Tuesday.
```

Feel free to adjust it to fit your specific needs.

## Notes

- **Error Handling**: The script includes basic error handling for token counting and file operations.
- **LLM Configuration**: You can modify the LLM initialization to use other supported models or APIs.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- Built with [LangChain](https://www.langchain.com/).
- Uses [Ollama LLM](https://www.ollama.ai/) for text summarization.

