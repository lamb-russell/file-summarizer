#!/usr/bin/env python3

import argparse
import os
from idlelib.pyparse import trans

from langchain_ollama.llms import OllamaLLM
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import tiktoken


def count_tokens(text, encoding_name="cl100k_base"):
    """Counts the number of tokens in the given text."""
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(text)
        return len(tokens)
    except Exception as e:
        print(f"Error counting tokens: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description='Summarize a text file (meeting transcript) using an LLM.')
    parser.add_argument('file_path', help='Path to the text file containing the meeting transcript.')
    args = parser.parse_args()

    transcript_text=None
    # Read the contents of the text file
    with open(args.file_path, 'r', encoding='utf-8') as f:
        transcript_text = f.read()

    # print(transcript_text)
    # Count tokens in the transcript
    token_count = count_tokens(transcript_text)
    print(f"Token count of the transcript: {token_count}")

    # Build the prompt for this one PR
    prompt_template = """Summarize the key points of this text as bullet points.  Who said what?  
    What were the most important parts of what was said?  What promises were made or follow ups needed?
    
    Example of good summary:
    1. Tom from product and Jerry from engineering met to discuss project "Mallet", a plan to design a better mouse trap
    2. Tom insisted Jerry complete the project next Tuesday.  Jerry proposed Thursday instead. 
    3. Jerry promised to follow up with Tom about project Mallet next tuesday.
    
    Text to process:
    {text_to_process}
    
    
    """

    prompt = PromptTemplate(
        input_variables=["text_to_process"],
        template=prompt_template
    ).format(text_to_process=transcript_text)

    # Initialize your LLM
    # Swap out OpenAI for another LLM if desired (e.g., HuggingFaceHub, Cohere, etc.)
    # llm = OpenAI(temperature=0)
    llm=OllamaLLM(model="mistral",num_predict=-2,num_ctx=32768)
    summary=llm(prompt=prompt)

    # Print the summary
    print("=== SUMMARY OF KEY POINTS ===")
    print(summary)

if __name__ == '__main__':
    main()
