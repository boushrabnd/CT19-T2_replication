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
- **results/**: Directory where prediction results are saved.
- **train/**: Scripts for training and ranking.
  - **rank_pages_bert.py**: Script to rank pages using AraBERT embeddings.

## Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/boushrabnd/CT19-T2_replication.git
   cd CT19-T2_replication
