# CT19-T2 Replication

This repository contains the replication code for Task 2 of the CLEF 2019 CheckThat! Lab, which focuses on ranking web pages based on their relevance to specific claims.

## Repository Structure

- **data/**: Contains the datasets used for training and evaluation.
  - **claims/**: JSON files with claims and associated web pages.
  - **html_pages/**: HTML files of the web pages.
  - **pages/**: CSV files with page usefulness labels.
- **data_processing/**: Scripts for preprocessing data.
  - **preprocessor.py**: Functions for cleaning and processing text data.
- **models/**: Directory for storing trained models.
- **results/**: Directory where prediction results are saved for each subtask.
  - **A/**: Directory where subtask A prediction and evaluation results are saved (there will be a folder for each subtask).
    - **predictions/**: Directory where prediction created by the runs are saved.
    - **evaluation/**: Directory evaluation results are saved.
- **train/**: Scripts for training and ranking.
  - **rank_pages_bert.py**: Script to rank pages using AraBERT embeddings.

## Dependencies

The code in this project uses Python 3.8 with the following key libraries:
- `torch` for deep learning (PyTorch)
- `transformers` for BERT and AraBERT models
- `beautifulsoup4` for HTML parsing
- `docx` for processing `.docx` files
- `pandas` and `nltk` for data manipulation and text processing

Install dependencies using:
```bash
pip install -r requirements.txt

## Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/boushrabnd/CT19-T2_replication.git
   cd CT19-T2_replication
