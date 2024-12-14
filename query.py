import open_clip
import torch
import json

# Load OpenCLIP model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model = model.to(device)
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# Load embeddings
with open("shot_embeddings.json", "r") as f:
    shot_embeddings = json.load(f)

# Normalize embeddings
for shot_name in shot_embeddings:
    shot_embeddings[shot_name] = torch.tensor(shot_embeddings[shot_name], device=device)
    shot_embeddings[shot_name] /= shot_embeddings[shot_name].norm(dim=-1, keepdim=True)

def retrieve_shots(query, shot_embeddings, top_k=4):
    """
    Retrieve and rank shots for a given query.
    """
    # Encode the query itself
    query_tokenized = tokenizer([query]).to(device)

    with torch.no_grad():
        # Encode and normalize the query.
        query_embedding = model.encode_text(query_tokenized)
        query_embedding /= query_embedding.norm(dim=-1, keepdim=True)

    results = []
    for shot_name, shot_embedding in shot_embeddings.items():
        similarity = (query_embedding @ shot_embedding.T).squeeze(0).item()
        results.append((shot_name, similarity))

    # Sort results by similarity
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

queries = [
    "A high-speed car chase",
    "An intense melee combat scene",
    "Character sobbing from an emotional breakdown",
    "Character being interrogated",
    "Two characters having a casual conversation",
    "A car driving normally or calmly",
    "Characters Climbing Large Structure"
]

output_file = "query_results.json"

# Retrieve results for each query and save them to a file
all_results = {}
for query in queries:
    results = retrieve_shots(query, shot_embeddings, top_k=4)
    all_results[query] = [{"shot": shot_name, "similarity": similarity} for shot_name, similarity in results]

# Save results to a JSON file
with open(output_file, "w") as f:
    json.dump(all_results, f, indent=4)

print(f"Results saved to {output_file}")
