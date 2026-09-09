import numpy as np
import torch
from tqdm import tqdm
from EmbGa import EmbeddingGA
import clip
import math
import utils
import time
from clip.simple_tokenizer import SimpleTokenizer
from utils import get_id_to_token_map

id_to_token = get_id_to_token_map()
output_directory = ""

def set_output_directory(directory):
    global output_directory
    output_directory = directory

def generate_candidate_token_ids(prompts, tokenizer):

    vocabulary_size = len(id_to_token)
    candidate_batches = []

    for prompt in prompts:
        candidate_tokens = torch.zeros((vocabulary_size, 77), dtype=torch.int32)
        prompt_tokens = tokenizer(prompt)[0]
        eos_positions = (prompt_tokens == 49407).nonzero(as_tuple=True)[0][0]
        prompt_tokens = prompt_tokens[:eos_positions]  # Remove tokens after the end-of-sequence token.
        prompt_size = prompt_tokens.shape[0]

        candidate_tokens[:, :prompt_size] = prompt_tokens
        candidate_tokens[:, prompt_size] = 267  # Comma token ID.
        for token_id in range(vocabulary_size):
            candidate_tokens[token_id, prompt_size + 1] = token_id

        candidate_tokens[:, prompt_size + 2] = 49407  # End-of-sequence token ID.

        candidate_batches.append(candidate_tokens)
    
    return torch.concatenate(candidate_batches)

def brute_force_search(prompts, images, top_k, model, tokenizer, file_prefix='', save_log=False):

    text_tokens = generate_candidate_token_ids(prompts, tokenizer)
    results = []
    
    # Tokenize the text and calculate the scores.
    scores = model(text_tokens, images).sum(dim=0)
    top_scores, top_indices = scores.topk(top_k)

    for score, candidate_id in zip(top_scores.numpy(), top_indices.numpy()):
        current_tokens = text_tokens[candidate_id]
        eos_index = (current_tokens == 49407).nonzero(as_tuple=True)[0][0].item()

        results.append((id_to_token[current_tokens[eos_index - 1].item()], prompts[candidate_id // len(id_to_token)], score.item(), candidate_id.item()))

    if save_log:
        utils.save_json(f'{output_directory}{file_prefix}_BF_log_{top_k}_{time.time()}.json', results)

    return results

def embedding_ga_search(prompts, device, images, iterations, biased_token_count, candidates, model, text_encoder, batch_comparator, tokenizer, niching=None, elite=False, comma=True, file_prefix='', save_log=False, mutation=True, mutation_rate=0.10, discrete_crossover=False, discrete_mutation=False):
    
    assert len(prompts) == 1, "EmbeddingGA currently supports only one prompt"
    simple_tokenizer = SimpleTokenizer()

    elite_tokens = None

    if elite:
        elite_tokens = [token_id for _, _, _, token_id in brute_force_search(prompts, images, biased_token_count, batch_comparator, tokenizer)]
        elite_tokens = torch.tensor(elite_tokens, dtype=torch.int32).to(device)
        elite_tokens = elite_tokens.view(1, -1)  # Reshape to (1, biased_token_count).

    
    ga = EmbeddingGA(
        population_size=candidates, 
        mutation_rate=mutation_rate, 
        prompt=prompts[0], 
        biased_token_count=biased_token_count,
        image_tensor=images,
        tokenizer=tokenizer, 
        model=batch_comparator,
        text_encoder=text_encoder,
        embedding_matrix=model.token_embedding.weight.detach().clone(),
        embedding=model.token_embedding, 
        device=device,
        comma=comma,
        elite_tokens=elite_tokens,
        niching=niching,
        mutation=mutation,
        discrete_crossover=discrete_crossover,
        discrete_mutation=discrete_mutation
        )

    predicted_tokens, history = ga.evolve(iterations)


    scores = batch_comparator(predicted_tokens, images).sum(dim=0)
    top_scores, top_indices = scores.topk(1)

    text_tokens = predicted_tokens[top_indices[0]].tolist()
    decoded_text = simple_tokenizer.decode(predicted_tokens[top_indices[0]].tolist())
    score = top_scores[0].item()

    if save_log:
        utils.save_json(f'{output_directory}{file_prefix}_GA_log_{niching}_{biased_token_count}_{iterations}_{candidates}_{time.time()}.json',
                   {
                        'text': decoded_text, 
                        'tokens': text_tokens,
                        'method': 'EmbGA',
                        'score': score,
                        'history': history
                    })

    return (decoded_text, text_tokens, score)
