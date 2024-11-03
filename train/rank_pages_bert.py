import pandas as pd
from data_processing.preprocessor import load_text_file, get_arabert_embedding
import os
import json
import torch

# Load claims
claims_file = 'data/claims/CT19-T2-Claims_with_Results.json'
with open(claims_file, 'r', encoding='utf-8') as f:
    claims_data = [json.loads(line) for line in f]

# Rank pages using BERT embeddings
def rank_pages_bert_embedding(claim_text, pages, claim_id):
    claim_embedding = get_arabert_embedding(claim_text)
    page_scores = []

    for page in pages:
        page_id = page.get("pageID")
        print(f"Processing claim {claim_id} and page {page_id}")
        
        # Try loading with different extensions
        for ext in [".html", ".txt", ".docx"]:
            filepath = f"data/html_pages/{page_id}{ext}"
            if os.path.exists(filepath):
                try:
                    page_text = load_text_file(filepath)
                    break  # Exit loop once the file is found and loaded
                except Exception as e:
                    print(f"Error loading file {filepath}: {e}")
                    continue
        else:
            print(f"No suitable file found for {page_id}")
            continue

        page_embedding = get_arabert_embedding(page_text)
        score = torch.nn.functional.cosine_similarity(
            torch.tensor(claim_embedding), torch.tensor(page_embedding), dim=0
        ).item()
        page_scores.append((page_id, score))

    # Sort pages by similarity score
    page_scores.sort(key=lambda x: x[1], reverse=True)
    return page_scores

# Process each claim and rank pages
ranked_results = []
run_id = "arabert_run"

for claim in claims_data:
    claim_id = claim['claimID']
    claim_text = claim['text']
    pages = claim['results']
    
    ranked_pages = rank_pages_bert_embedding(claim_text, pages, claim_id)
    
    for rank, (page_id, score) in enumerate(ranked_pages):
        ranked_results.append([claim_id, rank + 1, page_id, score, run_id])

# Save results to CSV
print("Saving results to CSV")
ranked_results_df = pd.DataFrame(ranked_results, columns=['claimID', 'rank', 'pageID', 'score', 'runID'])
ranked_results_df.to_csv('results/predictions_arabert.csv', index=False, sep='\t', header=False)
