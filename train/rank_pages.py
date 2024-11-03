import json
import pandas as pd

# load claims and page usefulness data
claims_file = 'data/claims/CT19-T2-Claims_with_Results.json'
usefulness_file = 'data/pages/CT19-T2-Pages_Usefulness.csv'

# load claims
with open(claims_file, 'r', encoding='utf-8') as f:
    claims_data = [json.loads(line) for line in f]

# load usefulness data
usefulness_df = pd.read_csv(usefulness_file, header=None, names=['claimID', 'pageID', 'label'])

# function to rank pages based on a reweighted usefulness score
def rank_pages_for_claim(claim_id, claim_pages):
    # filter usefulness data for the current claim
    relevant_pages = usefulness_df[usefulness_df['claimID'] == claim_id]
    
    # define weights for usefulness labels
    weights = {-1: 0.5, 0: 1.0, 1: 1.5, 2: 2.0}  
    
    # merge with claim pages and apply the weighting
    ranked_pages = []
    for idx, page in enumerate(claim_pages):
        page_id = page['pageID']
        # get usefulness label (-1: not relevant, 0: not useful, 1: useful, 2: very useful)
        label = relevant_pages[relevant_pages['pageID'] == page_id]['label'].values
        if len(label) > 0:
            score = weights.get(label[0], 0)  # Apply weight based on label
            # append the correct suffix for each page (idx + 1 as the suffix)
            ranked_pages.append((page_id, score))
    
    # sort pages by the weighted usefulness score
    ranked_pages.sort(key=lambda x: x[1], reverse=True)  # Higher score first
    return ranked_pages

# process each claim and rank its pages
ranked_results = []
run_id = 'teamXrun1'  # Specify the run ID for this result

for claim in claims_data:
    claim_id = claim['claimID']
    pages = claim['results']  # List of web pages for this claim
    ranked_pages = rank_pages_for_claim(claim_id, pages)
    
    # append ranked results to the list in the required format
    for rank, (page_id, score) in enumerate(ranked_pages):
        ranked_results.append([claim_id, rank + 1, page_id, score, run_id])  

# saving results
ranked_results_df = pd.DataFrame(ranked_results, columns=['claimID', 'rank', 'pageID', 'score', 'runID'])
ranked_results_df.to_csv('results/pred_A_run1.csv', index=False, sep='\t', header=False)
