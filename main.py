import os
import math
import random
import ir_datasets
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from itertools import islice
import torch
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import textwrap

experiment_doc_count = 1000000
experiment_query_count = 1000000

training_split = 0.8
ridge_alpha = 2

output_path = "./results"

tokenizer_name = "meta-llama/Llama-2-7b"
base_model_name = "meta-llama/Llama-2-7b-hf"
model_name = "castorini/rankllama-v1-7b-lora-passage"

def compute_collection_stats(tokenizer, collection_iter):
    sample_count = 0
    term_count = 0
    frequency_table = Counter()
    occurrence_table = Counter()

    for item in collection_iter:
        tokens = tokenizer(item.text).input_ids

        sample_count += 1
        term_count += len(tokens)

        frequency_table.update(set(tokens))
        occurrence_table.update(tokens)

    return sample_count, term_count, frequency_table, occurrence_table

def compute_inverse_score(token_ids, table, total):
    return np.array([
        [
            math.log(total / (table.get(token_id, 0) + 1))
        ]
        for token_id in token_ids
    ], dtype=np.float32)

def all_inverse_scores(token_ids, statistics):
    inverse_scores = { name: compute_inverse_score(token_ids, table, total) for name, table, total in statistics }

    inverse_scores["Random Noise"] = np.random.rand(len(token_ids), 1)

    return inverse_scores

def compute_embeddings(token_ids, model):
    input_embeddings = model.model.embed_tokens(torch.tensor(token_ids))

    normalized_input_embeddings = model.model.layers[0].input_layernorm(input_embeddings)

    qmat_embeddings = model.model.layers[0].self_attn.q_proj(normalized_input_embeddings)
    kmat_embeddings = model.model.layers[0].self_attn.k_proj(normalized_input_embeddings)

    return {
        "Input Embeddings": input_embeddings.numpy(),
        "Query Matrix Embeddings": qmat_embeddings.numpy(),
        "Key Matrix Embeddings": kmat_embeddings.numpy()
    }

def compute_regression(training_x, training_y):
    regression = Ridge(alpha=ridge_alpha)
    regression.fit(training_x, training_y)

    return regression

def predict_inverse_scores_from_embeddings(token_set_name, vocab_size, inverse_scores, embeddings):
    results = { token_set_name: {} }

    training_samples = math.floor(vocab_size * training_split)

    scaler = StandardScaler()
    
    for score_key, score_value in inverse_scores.items():
        results[token_set_name][score_key] = {}

        for embedding_key, embedding_value in embeddings.items():
            training_x = scaler.fit_transform(embedding_value[:training_samples, :])
            validation_x = scaler.transform(embedding_value[training_samples:, :])

            training_y = score_value[:training_samples, :]
            validation_y = score_value[training_samples:, :]

            regression = compute_regression(training_x, training_y)

            training_y_pred = regression.predict(training_x)
            validation_y_pred = regression.predict(validation_x)

            training_r2 = r2_score(training_y, training_y_pred)
            validation_r2 = r2_score(validation_y, validation_y_pred)

            results[token_set_name][score_key][embedding_key] = {
                "Training": training_r2,
                "Validation": validation_r2
            }

            print(f"Predicting {score_key} from {embedding_key} of {token_set_name}")
            print(f"\tTraining Data rSquared: {training_r2}")
            print(f"\tValidation Data rSquared: {validation_r2}")

    return results

def plot(output_file, title, feature_names, feature_values):
    plt.clf()

    plt.figure(figsize=(16, 10))

    wrapped_labels = [
        "\n".join(textwrap.wrap(feature_name, 16))
        for feature_name in feature_names
    ]

    bars = plt.bar(wrapped_labels, feature_values)

    plt.ylim(0.0, 1.0)
    plt.ylabel("R Squared")
    plt.title(title, pad=20, fontsize=20)

    for bar, value in zip(bars, feature_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            max(value, 0) + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=14
        )

    plt.tight_layout()
    plt.savefig(f"{output_path}/{output_file}")
    plt.close()

def plot_results(results):
    pool_keys = [ "Doc Terms", "Query Terms", "Collection Terms" ]
    score_keys = [ "Inverse Document Frequency", "Inverse Document Occurrences", "Inverse Query Frequency", "Inverse Query Occurrences", "Inverse Collection Frequency", "Inverse Collection Occurrences", "Random Noise" ]
    embedding_keys = [ "Input Embeddings", "Query Matrix Embeddings", "Key Matrix Embeddings" ]
    split_keys = [ "Training", "Validation" ]

    # ex: docTerms/idf/training -> input
    for pool_key in pool_keys:
        for score_key in score_keys:
            for split_key in split_keys:
                os.makedirs(f"{output_path}/{pool_key}/{score_key}", exist_ok=True)

                plot(f"{pool_key}/{score_key}/{split_key}.png", f"Predicting {score_key} from Word Embeddings of {pool_key}", embedding_keys, [
                    results[pool_key][score_key][embedding_key][split_key] for embedding_key in embedding_keys
                ])

    # ex: queryTerms/input/validation -> idf
    for pool_key in pool_keys:
        for embedding_key in embedding_keys:
            for split_key in split_keys:
                os.makedirs(f"{output_path}/{pool_key}/{embedding_key}", exist_ok=True)

                plot(f"{pool_key}/{embedding_key}/{split_key}.png", f"Predicting Rarity Features from {embedding_key} of {pool_key}", score_keys, [
                    results[pool_key][score_key][embedding_key][split_key] for score_key in score_keys
                ])

    # ex: input/igo/train -> docTerms
    for embedding_key in embedding_keys:
        for score_key in score_keys:
            for split_key in split_keys:
                os.makedirs(f"{output_path}/{embedding_key}/{score_key}", exist_ok=True)

                plot(f"{embedding_key}/{score_key}/{split_key}.png", f"Predicting {score_key} from {embedding_key}", pool_keys, [
                    results[pool_key][score_key][embedding_key][split_key] for pool_key in pool_keys
                ])

def visualize_score_map(output_file, text, tokenizer, compute_scores, compute_score_predictions):
    tokens = tokenizer(text, return_offsets_mapping=True)

    scores = compute_scores(tokens["input_ids"])
    score_predictions = compute_score_predictions(tokens["input_ids"])

    plt.clf()

    fig, ax = plt.subplots()

    x = 0
    
    for token_index in range(len(tokens["input_ids"])):
        token_text = text[tokens["offset_mapping"][token_index][0]:tokens["offset_mapping"][token_index][1]]
        
        if token_text == "": continue
        
        text_obj = ax.text(x, 8.15, token_text, fontsize=18, va="bottom", ha='left')
        
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        
        bbox_pixels = text_obj.get_window_extent(renderer=renderer)
        x_pixels = ax.transData.transform([(x, 0)])[0][0]
        pixel_width = bbox_pixels.width
        x_end_pixels = x_pixels + pixel_width
        x_start_data = ax.transData.inverted().transform([(x_pixels, 0)])[0][0]
        x_end_data = ax.transData.inverted().transform([(x_end_pixels, 0)])[0][0]
        width = x_end_data - x_start_data
        
        token_score = round(scores[token_index], 3)
        normalized_token_score = max(min(token_score, 10.0), 0.0) / 10.0
        score_color = (normalized_token_score, 0.0, 0.0)
        ax.add_patch(patches.Rectangle(
            (x, 4),
            width,
            3.5,
            linewidth=0,
            facecolor=score_color
        ))
        
        token_score_prediction = round(float(score_predictions[token_index]), 3)
        normalized_token_score_prediction = max(min(token_score_prediction, 10.0), 0.0) / 10.0
        score_prediction_color = (0.0, 0.0, normalized_token_score_prediction)
        ax.add_patch(patches.Rectangle(
            (x, 0),
            width,
            3.5,
            linewidth=0,
            facecolor=score_prediction_color
        ))
        
        x += width
        
    fig.canvas.draw()
    current_fig_width = fig.get_figwidth()
    current_xlim = ax.get_xlim()
    coefficient = current_fig_width / (current_xlim[1] - current_xlim[0])

    fig.set_size_inches(x * coefficient, 2)

    ax.set_xlim(0, x)
    ax.set_ylim(0, 10)
    ax.axis("off")

    plt.savefig(f"{output_path}/{output_file}", dpi=100, bbox_inches="tight")

    plt.close()

def main():
    # load model and dataset

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=1)

    model = PeftModel.from_pretrained(base_model, model_name, torch_dtype=torch.float16)
    model = model.merge_and_unload()

    collection = ir_datasets.load("msmarco-passage/train")

    # counts/tables for docs, queries, and collection

    doc_sample_count, doc_term_count, doc_frequency_table, doc_occurrence_table = compute_collection_stats(tokenizer, islice(collection.docs_iter(), experiment_doc_count))
    query_sample_count, query_term_count, query_frequency_table, query_occurrence_table = compute_collection_stats(tokenizer, islice(collection.queries_iter(), experiment_query_count))

    collection_sample_count = doc_sample_count + query_sample_count
    collection_term_count = doc_term_count + query_term_count
    collection_frequency_table = doc_frequency_table + query_frequency_table
    collection_occurrence_table = doc_occurrence_table + query_occurrence_table

    doc_token_ids = [ token_id for token_id in doc_frequency_table.keys() ]
    query_token_ids = [ token_id for token_id in query_frequency_table.keys() ]
    collection_token_ids = [ token_id for token_id in collection_frequency_table.keys() ]

    random.seed(112358)
    random.shuffle(doc_token_ids)
    random.shuffle(query_token_ids)
    random.shuffle(collection_token_ids)

    print("Document Statistics: ")
    print(f"\tVocab Size: {len(doc_token_ids)}")

    print("Query Statistics: ")
    print(f"\tVocab Size: {len(query_token_ids)}")

    print("Collection Statistics: ")
    print(f"\tVocab Size: {len(collection_token_ids)}")

    # inverse df, do, qf, qo, cf, co for doc, query, and collection terms

    statistics = [
        [ "Inverse Document Frequency", doc_frequency_table, doc_sample_count ],
        [ "Inverse Document Occurrences", doc_occurrence_table, doc_term_count ],
        [ "Inverse Query Frequency", query_frequency_table, query_sample_count ],
        [ "Inverse Query Occurrences", query_occurrence_table, query_term_count ],
        [ "Inverse Collection Frequency", collection_frequency_table, collection_sample_count ],
        [ "Inverse Collection Occurrences", collection_occurrence_table, collection_term_count ]
    ]

    np.random.seed(9753)
    doc_term_inverse_scores = all_inverse_scores(doc_token_ids, statistics)
    query_term_inverse_scores = all_inverse_scores(query_token_ids, statistics)
    collection_term_inverse_scores = all_inverse_scores(collection_token_ids, statistics)

    # input, qmat, and kmat embeddings for docs, queries, and collection

    doc_term_embeddings = compute_embeddings(doc_token_ids, model)
    query_term_embeddings = compute_embeddings(query_token_ids, model)
    collection_term_embeddings = compute_embeddings(collection_token_ids, model)

    # compute prediction results

    doc_term_results = predict_inverse_scores_from_embeddings("Doc Terms", len(doc_token_ids), doc_term_inverse_scores, doc_term_embeddings)
    query_term_results = predict_inverse_scores_from_embeddings("Query Terms", len(query_token_ids), query_term_inverse_scores, query_term_embeddings)
    collection_term_results = predict_inverse_scores_from_embeddings("Collection Terms", len(collection_token_ids), collection_term_inverse_scores, collection_term_embeddings)

    results = doc_term_results | query_term_results | collection_term_results

    plot_results(results)

    icf_scaler = StandardScaler()
    icf_regression = compute_regression(
        icf_scaler.fit_transform(collection_term_embeddings["Input Embeddings"]),
        collection_term_inverse_scores["Inverse Collection Frequency"]
    )

    visualization_examples = [
        "Hello my name is Matt and I'm a programmer with an interest in computer science.",
        "Recently I've been working on mechanistic interpretability of neural reranking models.",
        "I wonder how well my ridge regression can predict the rarity of different tokens from their embeddings.",
        "A good way to see is by comparing features of tokens with esoteric latent space compositions."
    ]

    os.makedirs(f"{output_path}/Visualizations", exist_ok=True)

    for index, visualization_example in enumerate(visualization_examples, start=1):
        visualize_score_map(
            f"Visualizations/Example{index}.png",
            visualization_example,
            tokenizer,
            lambda token_ids: [ math.log(collection_sample_count / (collection_frequency_table.get(token_id, 0) + 1)) for token_id in token_ids ],
            lambda token_ids: icf_regression.predict(icf_scaler.transform(model.model.embed_tokens(torch.tensor(token_ids))))
        )

if __name__ == "__main__": main()