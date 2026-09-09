import torch
from tqdm import tqdm
from utils import find_l2_distances
import time

torch.set_printoptions(sci_mode=False)

class EmbeddingGA:
    def __init__(self, population_size: int, mutation_rate: float, prompt: str, biased_token_count: int, image_tensor, tokenizer, model, text_encoder, embedding_matrix, embedding, device: str, comma: bool = True, elite_tokens=None, niching=None, mutation=True, discrete_crossover=False, discrete_mutation=False):
        self.__mutation_rate = mutation_rate
        self.__biased_token_count = biased_token_count
        self.__comma = comma
        self.__niching = niching
        self.__mutation = mutation

        self.__model = model
        self.__text_encoder = text_encoder
        self.__embedding_matrix = embedding_matrix.half().to(device)
        self.__embedding = embedding
        self.__device = device
        self.__tokenizer = tokenizer

        self.__prompt = prompt
        self.__population, self.__mutable_range, self.__immutable_positions = self.__initialize_population(population_size, self.__biased_token_count, elite_tokens)
        self.__population_size = self.__population.shape[0]
        self.__image_tensor = image_tensor

        self.__discrete_crossover = discrete_crossover
        self.__discrete_mutation = discrete_mutation

    def __initialize_population(self, population_size: int, biased_token_count: int, elite_tokens=None):
        elite_size = elite_tokens.shape[0] if elite_tokens is not None else 0
        token_population = torch.zeros((population_size + elite_size, 77), dtype=torch.int32)

        prompt_tokens = self.__tokenizer(self.__prompt)[0]
        eos_positions = (prompt_tokens == 49407).nonzero(as_tuple=True)[0][0]
        prompt_tokens = prompt_tokens[:eos_positions]  # Remove tokens after the end-of-sequence token.
        prompt_size = prompt_tokens.shape[0]

        token_population[:, :prompt_size] = prompt_tokens
        immutable_positions = list(range(prompt_size))
        current_pos = prompt_size

        range_low = current_pos
        if elite_tokens is not None:
            assert elite_tokens.shape[1] == biased_token_count, "The number of elite tokens must match biased_token_count"

        for token_index in range(biased_token_count):

            if self.__comma:
                token_population[:, current_pos] = 267  # Comma token ID.
                immutable_positions.append(current_pos)
                current_pos += 1


            token_population[elite_size:, current_pos] = torch.randint(0, 49407, (population_size,), device=self.__device)  # Random token.

            if elite_tokens is not None:
                token_population[0:elite_size, current_pos] = elite_tokens[:, token_index]

            current_pos += 1
        
        range_high = current_pos
        token_population[:, current_pos] = 49407  # End-of-sequence token ID.
        immutable_positions.append(current_pos)
        immutable_positions.extend(range(current_pos + 1, 77))
        
        return token_population.to(self.__device), [range_low, range_high], torch.tensor(immutable_positions, device=self.__device)

    def __fitness(self, population):
        scores = self.__model(population, self.__image_tensor).sum(dim=0)
        return scores

    def __tournament_selection(self, fitness_scores, population, tournament_size):
        
        if tournament_size >= population.shape[0]:
            raise ValueError("tournament_size must be less than the population size")

        group_count = fitness_scores.shape[0] // tournament_size
        usable_size = group_count * tournament_size

        perm = torch.randperm(population.shape[0], device=self.__device)
        groups = perm[:usable_size].view(group_count, tournament_size)
        
        group_fitness = fitness_scores[groups]
        local_best = group_fitness.argmax(dim=1)

        winner_indices = groups[torch.arange(group_count, device=population.device), local_best]


        return population[winner_indices], winner_indices

    def __roulette_selection(self, fitness_scores, population, parent_count):
        
        if parent_count % 2 != 0:
            parent_count += 1  # Ensure an even number of parents for pairing.

        if parent_count > fitness_scores.shape[0]:
            raise ValueError("parent_count must be less than or equal to the population size")
        
        if parent_count == len(fitness_scores):
            return population
        
        # Use roulette-wheel selection.
        normalized_fitness = fitness_scores / fitness_scores.sum()
        selected_indices = torch.multinomial(normalized_fitness, parent_count, replacement=False)

        return population[selected_indices]
    
    def __token_ids_to_vectors(self, tokens):
        return self.__embedding(tokens)
    
    def __vectors_to_token_ids(self, vectors):
        BATCH_SIZE = 512

        token_batches = []

        for i in range(0, vectors.shape[0], BATCH_SIZE):
            j = min(i + BATCH_SIZE, vectors.shape[0])
            batch_vectors = vectors[i : j]

            distances = find_l2_distances(batch_vectors.half(), self.__embedding_matrix)
            distance_indices = distances.argmin(dim=1).to(torch.int32)  # [N]
            token_batches.append(distance_indices.view(batch_vectors.shape[0], batch_vectors.shape[1]))
        
        return torch.cat(token_batches, dim=0)
    
    def __crossover(self, selected_parent_vectors):
        # Perform crossover between two parents to create offspring.
        num_of_pairs = selected_parent_vectors.shape[0] // 2

        parent0 = selected_parent_vectors[0 : num_of_pairs]
        parent1 = selected_parent_vectors[num_of_pairs : num_of_pairs * 2]

        crossover_points0 = torch.rand((parent0.shape[0], parent0.shape[1], 1), device=self.__device)
        crossover_points1 = torch.rand((parent0.shape[0], parent0.shape[1], 1), device=self.__device)

        children0 = parent0 * crossover_points0 + parent1 * (1 - crossover_points0)
        children1 = parent0 * crossover_points1 + parent1 * (1 - crossover_points1)
        
        children = torch.cat([children0, children1], dim=0)

        return children

    def __discrete_crossover_tokens(self, selected_parents):
        num_of_pairs = selected_parents.shape[0] // 2

        parent0 = selected_parents[0:num_of_pairs]
        parent1 = selected_parents[num_of_pairs:num_of_pairs * 2]
        onepoint_crossover = torch.randint(self.__mutable_range[0], self.__mutable_range[1], (num_of_pairs,), device=self.__device)

        slot_ids = torch.arange(parent0.shape[1], device=self.__device).unsqueeze(0)
        mask = slot_ids < onepoint_crossover.unsqueeze(1)

        children0 = parent0 * mask + parent1 * ~mask
        children1 = parent1 * mask + parent0 * ~mask

        return torch.cat([children0, children1], dim=0)

    def __discrete_mutate(self, children):
        children_mutate_pct = torch.rand(children.shape[0], children.shape[1], device=self.__device)
        children_mutate_pct[:, self.__immutable_positions] = 1  # Never mutate immutable positions.
        children_mutate_mask = children_mutate_pct < self.__mutation_rate
        selected_children = children_mutate_mask.any(dim=1)
        children_mutate_mask = children_mutate_mask[selected_children]

        if children_mutate_mask.shape[0] == 0:
            return children

        random_tokens = self.__initialize_population(children_mutate_mask.shape[0], self.__biased_token_count)[0]
        children[selected_children] = (children[selected_children] * ~children_mutate_mask) + (random_tokens * children_mutate_mask)

        return children

    def __mutate(self, children_vectors):
        children_mutate_pct = torch.rand(children_vectors.shape[0], children_vectors.shape[1], device=self.__device)
        children_mutate_pct[:, self.__immutable_positions] = 1  # Never mutate immutable positions.
        children_mutate_mask = children_mutate_pct < self.__mutation_rate
        selected_children = children_mutate_mask.any(dim=1)
        children_mutate_mask = children_mutate_mask[selected_children]

        if children_mutate_mask.shape[0] == 0:
            return children_vectors

        random_vectors = self.__token_ids_to_vectors(self.__initialize_population(children_mutate_mask.shape[0], self.__biased_token_count)[0])
        interpolation_weights = torch.rand((children_mutate_mask.shape[0], children_mutate_mask.shape[1], 1), device=self.__device)
        interpolation_weights = interpolation_weights * children_mutate_mask.unsqueeze(-1)  # Never mutate unselected positions.

        children_vectors[selected_children] = children_vectors[selected_children] * (1 - interpolation_weights) + random_vectors * interpolation_weights

        return children_vectors

    def __crowding(self, population, parents, parent_indices, parent_fitness, children, children_fitness):
        pair_count = parents.shape[0] // 2
        parent_encoding_vectors = self.__text_encoder(parents).to(self.__device)  # [B, L]
        child_encoding_vectors = self.__text_encoder(children).to(self.__device)  # [B, L]
        parent0_fitness = parent_fitness[0:pair_count]
        parent1_fitness = parent_fitness[pair_count:pair_count * 2]
        parent0_vectors = parent_encoding_vectors[0:pair_count]
        parent1_vectors = parent_encoding_vectors[pair_count:pair_count * 2]
        parent0_indices = parent_indices[0:pair_count]
        parent1_indices = parent_indices[pair_count:pair_count * 2]

        children0 = children[0:pair_count]
        children1 = children[pair_count:pair_count * 2]
        children0_fitness = children_fitness[0:pair_count]
        children1_fitness = children_fitness[pair_count:pair_count * 2]
        children0_vectors = child_encoding_vectors[0:pair_count]
        children1_vectors = child_encoding_vectors[pair_count:pair_count * 2]
        
        # D(p0, c0) + D(p1, c1)
        dist0 = ((parent0_vectors - children0_vectors) ** 2).sum(dim=1) + ((parent1_vectors - children1_vectors) ** 2).sum(dim=1)
        # D(p0, c1) + D(p1, c0)
        dist1 = ((parent0_vectors - children1_vectors) ** 2).sum(dim=1) + ((parent1_vectors - children0_vectors) ** 2).sum(dim=1)

        # Pair each child with the more similar parent, then retain the fitter individual.
        direct_pairing_mask = dist0 <= dist1
        child0_beats_parent0 = direct_pairing_mask & (children0_fitness > parent0_fitness)
        population[parent0_indices[child0_beats_parent0]] = children0[child0_beats_parent0]
        child1_beats_parent1 = direct_pairing_mask & (children1_fitness > parent1_fitness)
        population[parent1_indices[child1_beats_parent1]] = children1[child1_beats_parent1]
        crossed_pairing_mask = ~direct_pairing_mask
        child1_beats_parent0 = crossed_pairing_mask & (children1_fitness > parent0_fitness)
        population[parent0_indices[child1_beats_parent0]] = children1[child1_beats_parent0]
        child0_beats_parent1 = crossed_pairing_mask & (children0_fitness > parent1_fitness)
        population[parent1_indices[child0_beats_parent1]] = children0[child0_beats_parent1]
        


    def __culling(self, population, population_to_keep):
        # Evaluate fitness scores and select the top-k individuals.
        fitness_scores = self.__fitness(population)
        topk_scores, topk_indices = fitness_scores.topk(population_to_keep, dim=0)

        return population[topk_indices], topk_scores

    def evolve(self, num_generations:int):
        history = []

        with torch.no_grad():
            for generation in tqdm(range(num_generations)):

                # Evaluate fitness scores for the current population.
                fitness_scores = self.__fitness(self.__population).to(self.__device)

                # Select parents based on their fitness scores.
                if self.__niching is None:
                    parents = self.__roulette_selection(fitness_scores, self.__population, self.__population.shape[0] // 2)

                if self.__niching == "crowding":
                    parents, parent_indices = self.__tournament_selection(fitness_scores, self.__population, tournament_size=3)

                # Perform crossover to generate children.
                children_vectors = None

                if self.__discrete_crossover:
                    children = self.__discrete_crossover_tokens(parents)
                    children_vectors = self.__token_ids_to_vectors(children)
                else:
                    parent_vectors = self.__token_ids_to_vectors(parents)
                    children_vectors = self.__crossover(parent_vectors)

                # Mutate the children.
                if self.__mutation:
                    if self.__discrete_mutation:
                        children = self.__vectors_to_token_ids(children_vectors)
                        children = self.__discrete_mutate(children)
                        children_vectors = self.__token_ids_to_vectors(children)
                    else:
                        children_vectors = self.__mutate(children_vectors)


                # Convert the child vectors back to token IDs.
                children = self.__vectors_to_token_ids(children_vectors)
                
                # Eliminate less-fit individuals.
                if self.__niching == "crowding":
                    self.__crowding(self.__population, 
                                    parents, parent_indices,
                                    fitness_scores[parent_indices], children, self.__fitness(children).to(self.__device))
                else:
                    self.__population, _ = self.__culling(
                        torch.cat([self.__population, children], dim=0),
                        self.__population_size)

                
                topk_scores, _ = fitness_scores.topk(1, dim=0)
                history.append({'iter': generation, 'score': topk_scores[0].item(), 'time': time.time()})

        return self.__population, history
