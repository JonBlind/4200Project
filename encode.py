import open_clip
import torch
import os
import json
from PIL import Image

'''
Due to short time and being a little lost as to how I can interact with openCLIP
I did utilize AI for help in creating this script; though, I made sure I understood
what was happening before continuing.
'''
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model = model.to(device) 

def encode_shot(shot_folder):
    """
    Encode all frames present in their respective shot folders using OpenCLIP.
    """
    frame_embeddings = []

    # for every file in the shot_folder, if the file ends with a .jpg, we
    # encode it and add it to our frame embeddings.
    for filename in sorted(os.listdir(shot_folder)):
        if filename.endswith(('.jpg')):
            frame_path = os.path.join(shot_folder, filename)
            image = preprocess(Image.open(frame_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                frame_embedding = model.encode_image(image)
            frame_embeddings.append(frame_embedding)

    if not frame_embeddings:
        raise ValueError(f"No frames found in folder: {shot_folder}")
    
    # Aggregate embeddings. Essentially, we average out the information present in the shots.
    return torch.mean(torch.stack(frame_embeddings), dim=0).cpu().numpy()

def process_shots(shots_folder, output_file):
    """
    Process all shots using the encoding method where we encode and aggregate all shots.
    """
    shot_embeddings = {}

    # For every folder in the shots folder, we append the folder names to the end of the path to access them.
    # Which we then pass all the contents into "encode_shot"
    for folder in sorted(os.listdir(shots_folder)):
        shot_folder = os.path.join(shots_folder, folder)
        if os.path.isdir(shot_folder):
            print(f"Encoding shot: {folder}")
            try:
                shot_embeddings[folder] = encode_shot(shot_folder).tolist()
            except ValueError as e:
                print(e)
    
    # Save embeddings to a JSON file
    with open(output_file, "w") as f:
        json.dump(shot_embeddings, f, indent=4)
    print(f"Shot embeddings saved to {output_file}")


shots_folder = "C:/Users/prepr/Desktop/School/InfoRetrieval/Shots"
output_file = "shot_embeddings.json"

process_shots(shots_folder, output_file)
