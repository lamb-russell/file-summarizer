#!/usr/bin/env python3
"""
Summary Script for Meeting Transcripts using LLM. This script reads through text files (meeting transcripts), counts the number of tokens, 
and generates a summary for each file using an Language Model (LLM). It can process either a single file or multiple files within a specified directory.

Features:
   - Token counting for text files
   - Generating summaries from transcripts using LLM
   - Handling directories with modified files in the last 'n' days

Usage:
   To process a single text file, run `python main.py <file_path>`.
   To process multiple text files within a directory, run `python main.py --directory <directory_path>` and specify the number of days to look back with `--days <n>`.

Requirements:
   - Python 3
   - Tiktoken library for token counting
   - Langchain and Langchain-ollama libraries for LLM usage
"""

import argparse
import datetime
import os

import tiktoken
from langchain.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM


def count_tokens(text, encoding_name="cl100k_base"):
    """Counts the number of tokens in the given text."""
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(text)
        return len(tokens)
    except Exception as e:
        print(f"Error counting tokens: {e}")
        return 0

def get_files_modified_last_n_days(directory, days):
    """Get a list of files in the directory modified in the last n days."""
    last_n_days_files = []
    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)

    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                file_modified_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
                if file_modified_time >= cutoff_date:
                    last_n_days_files.append(file_path)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    return last_n_days_files

def process_file(file_path):
    """Process a single file and summarize its content."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            transcript_text = f.read()

        # Count tokens in the transcript
        token_count = count_tokens(transcript_text)
        print(f"Token count of {file_path}: {token_count}")

        # Build the prompt
        prompt_template = """You're an expert executive assistant.  
        Summarize the key points of this transcript as bullet points for executives.  What is the context? 
        Who is involved?  What was said?  What promises were made or follow ups needed?
        
        Text to process:
        {text_to_process}
        """

        prompt = PromptTemplate(
            input_variables=["text_to_process"],
            template=prompt_template
        ).format(text_to_process=transcript_text)

        # Initialize your LLM
        llm = OllamaLLM(model="mistral", num_predict=-2, num_ctx=32768)
        summary = llm(prompt=prompt)

        # Print the summary
        print(f"=== SUMMARY OF KEY POINTS FOR {file_path} ===")
        print(summary)

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Summarize text files (meeting transcripts) using an LLM.')
    parser.add_argument('file_path', nargs='?', help='Path to a single text file containing the meeting transcript.')
    parser.add_argument('--directory', help='Path to a directory containing text files to process.')
    parser.add_argument('--days', type=int, default=7, help='Number of days to look back for modified files in the directory.')
    args = parser.parse_args()

    if args.directory:
        files_to_process = get_files_modified_last_n_days(args.directory, args.days)
        print(f"Found {len(files_to_process)} files modified in the last {args.days} days.")

        for file_path in files_to_process:
            process_file(file_path)
    elif args.file_path:
        process_file(args.file_path)
    else:
        print("Please provide either a file path or a directory path to process.")

if __name__ == '__main__':
    main()
