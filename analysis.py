import json
import pandas as pd
import numpy as np

# Load query results
with open("query_results.json", "r") as f:
    query_results = json.load(f)

# Load relevance judgments
judgments = pd.read_csv("judgments.csv")
judgments.set_index("Unnamed: 0", inplace=True)

def precision_k(retrieved, relevance_column, k):
    """
    For a given query, calculate precision@k.
    Essentially, calculate the mean of shot results being relevant.
    """
    relevant_count = 0
    for shot in retrieved[:k]:
        if shot in relevance_column.index and relevance_column[shot] > 0:
            relevant_count += 1
    return relevant_count / k

def ndcg_k(retrieved, relevance_column, k):
    """
    For a given query, calculate NDCG@K
    """
    dcg = 0.0
    for i, shot in enumerate(retrieved[:k]):
        if shot in relevance_column.index:
            rel = relevance_column[shot]
            dcg += (2 ** rel - 1) / np.log2(i + 2)

    # Ideal DCG
    ideal_relevance = relevance_column.sort_values(ascending=False)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_relevance):
        idcg += (2 ** rel - 1) / np.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0



results_summary = []
for query in query_results:
    retrieved_shots = [result["shot"] for result in query_results[query]]

    # Get relevance judgments for this query
    if query in judgments.columns:
        relevance_column = judgments[query]
    else:
        print(f"Warning: No judgments found for query '{query}'")
        relevance_column = pd.Series(dtype=int)

    # Calculate metrics
    precision = precision_k(retrieved_shots, relevance_column, k=4)
    ndcg = ndcg_k(retrieved_shots, relevance_column, k=4)

    # Append to results summary
    results_summary.append({"Query": query, "Precision@4": precision, "NDCG@4": ndcg})


# Convert results summary to DataFrame for easy viewing
results_df = pd.DataFrame(results_summary)
results_df.to_csv("evaluation_metrics.csv", index=False)
print("Evaluation metrics saved to evaluation_metrics.csv")