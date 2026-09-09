import torch
import os
from tqdm import tqdm
from PIL import Image
from clip.simple_tokenizer import SimpleTokenizer
import json
import numpy as np

def get_id_to_token_map():
    tokenizer = SimpleTokenizer()
    vocab = tokenizer.encoder  # token → id

    id_to_token = {token_id: tokenizer.decode([token_id]) for token_id in vocab.values()}  # id → token

    return id_to_token

def get_vocabulary():
    tokenizer = SimpleTokenizer()
    vocab = tokenizer.encoder  # token → id
    return vocab

def load_images(model, preprocess, model_device, directory, batch_size=512, image_limit=None):
    print(f"Loading images from {directory}")
    image_files = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith(('.png', '.jpg', '.jpeg'))]
    
    if image_limit is not None:
        image_files = image_files[:image_limit]

    image_features = []

    with torch.no_grad():
        for batch_start in tqdm(range(0, len(image_files), batch_size)):
            batch_end = min(batch_start + batch_size, len(image_files))
            batch_files = image_files[batch_start:batch_end]

            batch_images = [Image.open(filename) for filename in batch_files]
            batch_images = torch.stack([preprocess(image).to(model_device) for image in batch_images])
            batch_features = model.encode_image(batch_images)
            batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)

            image_features.append(batch_features.cpu())

    return torch.cat(image_features)

def create_batched_text_encoder(model, model_device, batch_size=512):

    def encode_text_batches(text_tokens):

        with torch.no_grad():

            text_count = text_tokens.shape[0]

            encoded_batches = []

            for batch_start in range(0, text_count, batch_size):
                batch_end = min(batch_start + batch_size, text_count)

                text_batch = text_tokens[batch_start:batch_end]
                text_features = model.encode_text(text_batch.to(model_device))
                
                encoded_batches.append(text_features.cpu())

            return torch.cat(encoded_batches, dim=0)
    
    return encode_text_batches

def create_batched_text_image_comparator(model, model_device, batch_size=512):
    
    def compare_text_and_images(text_tokens, image_tensor):

        with torch.no_grad():

            image_count = image_tensor.shape[0]
            device_image_tensor = image_tensor.to(model_device)
            text_count = text_tokens.shape[0]
            scale = model.logit_scale.exp()

            comparison_scores = torch.empty((image_count, text_count), dtype=torch.float32, device="cpu")

            for batch_start in range(0, text_count, batch_size):
                batch_end = min(batch_start + batch_size, text_count)

                text_batch = text_tokens[batch_start:batch_end]
                text_features = model.encode_text(text_batch.to(model_device))
                normalized_text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                logits_per_image = scale * (device_image_tensor @ normalized_text_features.T)
                comparison_scores[:, batch_start:batch_end] = logits_per_image.cpu()

            return comparison_scores
        
    return compare_text_and_images

def find_l2_distances(vectors, embeddings, batch_size=512):
    
    # Vectors have the shape (batch, sequence length, embedding dimension).
    batch_count, sequence_length, embedding_dimension = vectors.shape
    # Embeddings have the shape (vocabulary size, embedding dimension).
    vocabulary_size, stored_embedding_dimension = embeddings.shape

    assert embedding_dimension == stored_embedding_dimension, "Dimension mismatch between vectors and embeddings"

    # Flatten the vectors to shape (batch * sequence length, embedding dimension).
    flat_vectors = vectors.reshape(batch_count * sequence_length, embedding_dimension)

    # Calculate the vector and embedding norms.
    vector_norms = (flat_vectors ** 2).sum(dim=1, keepdim=True)  # (N, 1)
    embedding_norms = (embeddings ** 2).sum(dim=1, keepdim=True).T  # (1, V)

    # [N, V] = [N, 1] + [1, V] - 2 * [N, D] @ [D, V]
    distances = vector_norms + embedding_norms - 2 * flat_vectors @ embeddings.T

    return distances

def save_json(filename, data):
    print(filename)
    with open(filename, 'w') as f:
        json.dump(data, f)
