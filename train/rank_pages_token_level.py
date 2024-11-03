import os
import json
import pandas as pd
from bs4 import BeautifulSoup
from collections import Counter
import stanza

stanza.download('ar')

nlp = stanza.Pipeline('ar', processors='tokenize')

claims_file = 'data/claims/CT19-T2-Claims_with_Results.json'
html_dir = 'html_pages'
run_id = 'teamXrun2'  

with open(claims_file, 'r', encoding='utf-8') as f:
    claims_data = [json.loads(line) for line in f]

# extracting text content from HTML file
def extract_text_from_html(html_file_path):
    with open(html_file_path, 'r', encoding='utf-8') as html_file:
        print(html_file_path)
        soup = BeautifulSoup(html_file, 'html.parser')
        text = soup.get_text(separator=' ')
    return text

# preprocessing 
def preprocess_text_arabic(text):
    doc = nlp(text)
    tokens = [word.text for sentence in doc.sentences for word in sentence.words]
    return tokens

# ranking based on token-level matching
def rank_pages_token_level_arabic(claim_text, pages, claim_id):
    claim_tokens = preprocess_text_arabic(claim_text)
    page_scores = []

    for page in pages:
        page_id = page['pageID']
        html_file_path = os.path.join(html_dir, f"{page_id}.html")
        if os.path.exists(html_file_path):
            page_text = extract_text_from_html(html_file_path)
            page_tokens = preprocess_text_arabic(page_text)
            common_tokens = Counter(claim_tokens) & Counter(page_tokens)
            score = sum(common_tokens.values())
            page_scores.append((page_id, score))

    page_scores.sort(key=lambda x: x[1], reverse=True)
    return page_scores

# processing each claim and rank its pages
ranked_results = []
for claim in claims_data:
    claim_id = claim['claimID']
    claim_text = claim['text']
    pages = claim['results']  
    ranked_pages = rank_pages_token_level_arabic(claim_text, pages, claim_id)
    
    # appening ranked results to the list in the required format
    for rank, (page_id, score) in enumerate(ranked_pages):
        ranked_results.append([claim_id, rank + 1, page_id, score, run_id])  # claimID, rank, pageID, score, runID

# saving results
ranked_results_df = pd.DataFrame(ranked_results, columns=['claimID', 'rank', 'pageID', 'score', 'runID'])
ranked_results_df.to_csv('results/pred_A_run2.csv', index=False, sep='\t', header=False)
