import os
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModel
import torch
from docx import Document

def clean_html(content):
    """Extract and clean text from HTML content."""
    soup = BeautifulSoup(content, "html.parser")
    return soup.get_text(separator=" ")

def load_text_file(filepath):
    """Load content based on file type."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".html":
        with open(filepath, "r", encoding="utf-8") as f:
            return clean_html(f.read())
    elif ext == ".txt":
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    elif ext == ".docx":
        doc = Document(filepath)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def preprocess_text(text):
    """Tokenize and prepare text for Arabert model."""
    tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")
    tokens = tokenizer.tokenize(text)
    return tokens

def get_arabert_embedding(text):
    """Generate embeddings using AraBERT."""
    tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")
    model = AutoModel.from_pretrained("aubmindlab/bert-base-arabertv2")
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
