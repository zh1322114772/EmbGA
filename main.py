from algs import embedding_ga_search
import torch
import clip
import utils

def main():

    prompt = 'Image of a Chinese person eating noodles at a restaurant'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model = model.float() 

    image_vectors = utils.load_images(model, preprocess, device, "./Data/", batch_size=16)
    print(image_vectors.shape)
    batch_comparator = utils.create_batched_text_image_comparator(model, device, batch_size=16)
    text_encoder = utils.create_batched_text_encoder(model, device, batch_size=16)

    generated_text, _, generated_text_score = embedding_ga_search(
                [prompt], 
                device, 
                image_vectors, 
                iterations=250, 
                biased_token_count=10,
                candidates=3000, 
                model=model, 
                text_encoder=text_encoder, 
                batch_comparator=batch_comparator, 
                tokenizer=clip.tokenize, 
                niching='crowding')

    print(f"CLIP score: {generated_text_score / (image_vectors.shape[0] * 100)}")
    print(f"Text: {generated_text}")

if __name__ == "__main__":
    main()
