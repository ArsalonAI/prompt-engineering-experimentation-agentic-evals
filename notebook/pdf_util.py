import PyPDF2
import tiktoken
import re

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
        return text
    
def count_tokens(text):
    """
    Count tokens in text using tiktoken's cl100k_base encoder (GPT-4 style tokenization)
    
    Args:
        text (str): The input text to tokenize
        
    Returns:
        int: Number of tokens in the text
    """
    encoder = tiktoken.get_encoding("cl100k_base")
    return len(encoder.encode(text))


def extract_sections(text, section_headers=None):
    """
    Extracts sections from a given text based on provided section headers.

    Args:
        text (str): The full text of the research paper.
        section_headers (list, optional): A list of section header names to search for.
            Defaults to ['Abstract', 'Introduction', 'Methods', 'Conclusion'].

    Returns:
        dict: A dictionary mapping each header (capitalized) to its corresponding text content.
    """
    if section_headers is None:
        section_headers = ['Abstract', 'Introduction', 'Methods', 'Conclusion']
    
    # Build a regex pattern to match any of the provided section headers starting at the beginning of a line.
    # re.escape is used to safely handle any special characters in the headers.
    pattern = r'^(?P<header>' + "|".join(map(re.escape, section_headers)) + r')\b'
    header_pattern = re.compile(pattern, flags=re.IGNORECASE | re.MULTILINE)
    
    # Find all header matches in the text
    matches = list(header_pattern.finditer(text))
    
    # Dictionary to store the extracted sections
    sections = {}
    
    if not matches:
        # If no sections are found, you can decide how to handle this situation (e.g., return empty dict)
        return sections

    # Extract each section's text by using the start index of each match and the next one as delimiter.
    for i, match in enumerate(matches):
        # Get the header name and standardize it by capitalizing
        header = match.group("header").capitalize()
        # Start of current section: immediately after the header
        start = match.end()
        # End of the current section: start of next header or end of text
        end = matches[i + 1].start() if (i + 1) < len(matches) else len(text)
        # Capture the section text and strip leading/trailing whitespace
        sections[header] = text[start:end].strip()
    
    return sections

