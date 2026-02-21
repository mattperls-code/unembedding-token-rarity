import os
import math
import random
import ir_datasets
from collections import Counter
import nltk
import string
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from itertools import islice
import torch
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import textwrap

collection_size = 5000000
training_split = 0.8

pca_count = 400

ridge_alpha = 2.0
pca_ridge_alpha = 0

logistic_c = 0.1
pca_logistic_c = 1000

output_path = "./results"

tokenizer_name = "meta-llama/Llama-2-7b"
base_model_name = "meta-llama/Llama-2-7b-hf"
model_name = "castorini/rankllama-v1-7b-lora-passage"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=1)

model = PeftModel.from_pretrained(base_model, model_name, torch_dtype=torch.float16)
model = model.merge_and_unload()

collection = ir_datasets.load("msmarco-passage/train")

doc_count = 0
token_count = 0
frequency_table = Counter()
occurrence_table = Counter()

for doc in islice(collection.docs_iter(), collection_size):
    tokens = tokenizer(doc.text).input_ids

    doc_count += 1
    token_count += len(tokens)

    frequency_table.update(set(tokens))
    occurrence_table.update(tokens)

tokens = [ token for token in frequency_table.keys() ]

continuous_features = [
    "Inverse Frequency",        # |C| / (freq + 1)
    "Inverse Occurrences",      # total_count / (term_count + 1)
    "Log Inverse Frequency",    # log ( |C| / (freq + 1) )
    "Log Inverse Occurrences",  # log ( total_count / (term_count + 1) )
    "Random Noise"
]

logistic_features = [
    "Is Stopword",
    "Is Punctuation",
    "Random Class"
]

def inverse_frequency(token):
    return doc_count / (frequency_table.get(token, 0) + 1)

def inverse_occurrences(token):
    return token_count / (occurrence_table.get(token, 0) + 1)

def log_inverse_frequency(token):
    return math.log(doc_count / (frequency_table.get(token, 0) + 1))

def log_inverse_occurrences(token):
    return math.log(token_count / (occurrence_table.get(token, 0) + 1))

def random_noise(token):
    return random.random()

nltk.download("stopwords")

stop_tokens = set()

for stopword in nltk.corpus.stopwords.words("english"):
    for stop_token in islice(tokenizer(stopword).input_ids, 1, None):
        stop_tokens.add(stop_token)

def is_stopword(token):
    return 1 if token in stop_tokens else 0

punctuation_tokens = set([ tokenizer(char).input_ids[1] for char in string.punctuation ])

def is_punctuation(token):
    return 1 if token in punctuation_tokens else 0

def random_class(token):
    return 1 if random.random() > 0.5 else 0

compute_continuous_feature = {
    "Inverse Frequency": inverse_frequency,
    "Inverse Occurrences": inverse_occurrences,
    "Log Inverse Frequency": log_inverse_frequency,
    "Log Inverse Occurrences": log_inverse_occurrences,
    "Random Noise": random_noise
}

compute_logistic_feature = {
    "Is Stopword": is_stopword,
    "Is Punctuation": is_punctuation,
    "Random Class": random_class
}

random.seed(10203040)
random.shuffle(tokens)

continuous_feature_scores = np.array([
    [ compute_continuous_feature[feature_name](token) for token in tokens ]
    for feature_name in continuous_features
])

logistic_feature_scores = np.array([
    [ compute_logistic_feature[feature_name](token) for token in tokens ]
    for feature_name in logistic_features
])

def visualize_score_map(output_file, text, tokenizer, compute_scores, compute_score_predictions, max_score):
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
        normalized_token_score = max(min(token_score, max_score), 0.0) / max_score
        score_color = (normalized_token_score, 0.0, 0.0)
        ax.add_patch(patches.Rectangle(
            (x, 4),
            width,
            3.5,
            linewidth=0,
            facecolor=score_color
        ))
        
        token_score_prediction = round(float(score_predictions[token_index]), 3)
        normalized_token_score_prediction = max(min(token_score_prediction, max_score), 0.0) / max_score
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

max_feature_scores = {
    "Inverse Frequency": 2500,
    "Inverse Occurrences": 5000,
    "Log Inverse Frequency": 12,
    "Log Inverse Occurrences": 18,
    "Random Noise": 1.0,
    "Is Stopword": 1.0,
    "Is Punctuation": 1.0,
    "Random Class": 1.0
}

visualization_examples = [
    "Hello my name is Matt and I'm a programmer with an interest in computer science.",
    "Recently I've been working on the mechanistic interpretability of neural reranking models.",
    "I wonder how well various regression methods can predict the rarity of different tokens from their embeddings.",
    "This example has many very common words that should be present in many of the training documents.",
    "This evaluation instance possesses a plethora of intermittently distributed esoteric tokens.",
    "This is an example with a lot of stop words in it, designed to see if some words are more likely to be dropped.",
    "I'd've put more contractions in this example, but I didn't because I couldn't think of any, so we'll have to see.",
    "Finally done! Now I just need punctuation - markings such as ., ,, ;, and : - which are designed to convey structure."
]

token_embeddings = model.model.embed_tokens(torch.tensor(tokens))

continuous_features_r2 = {
    "Training Data (Without PCA)": [],
    "Validation Data (Without PCA)": [],
    "Training Data (With PCA)": [],
    "Validation Data (With PCA)": []
}

for feature_index, feature_name in enumerate(continuous_features):
    print(f"Evaluating {feature_name}")

    scaler = StandardScaler()

    training_sample_count = round(training_split * len(tokens))

    training_x = scaler.fit_transform(token_embeddings[:training_sample_count])
    validation_x = scaler.transform(token_embeddings[training_sample_count:])

    training_y = continuous_feature_scores[feature_index, :training_sample_count]
    validation_y = continuous_feature_scores[feature_index, training_sample_count:]

    regression = Ridge(alpha=ridge_alpha)
    regression.fit(training_x, training_y)

    training_y_pred = regression.predict(training_x)
    validation_y_pred = regression.predict(validation_x)

    continuous_features_r2["Training Data (Without PCA)"].append(r2_score(training_y, training_y_pred))
    continuous_features_r2["Validation Data (Without PCA)"].append(r2_score(validation_y, validation_y_pred))

    os.makedirs(f"{output_path}/Visualizations/Without PCA/{feature_name}", exist_ok=True)

    for example_index, example_text in enumerate(visualization_examples):
        visualize_score_map(
            f"Visualizations/Without PCA/{feature_name}/example{example_index}.png",
            example_text,
            tokenizer,
            lambda tokens: [ compute_continuous_feature[feature_name](token) for token in tokens ],
            lambda tokens: regression.predict(scaler.transform(model.model.embed_tokens(torch.tensor(tokens)))),
            max_feature_scores[feature_name]
        )

    pca = PCA(n_components=pca_count)
    
    pca_training_x = pca.fit_transform(training_x)
    pca_validation_x = pca.transform(validation_x)

    pca_regression = Ridge(alpha=pca_ridge_alpha)
    pca_regression.fit(pca_training_x, training_y)

    pca_training_y_pred = pca_regression.predict(pca_training_x)
    pca_validation_y_pred = pca_regression.predict(pca_validation_x)

    continuous_features_r2["Training Data (With PCA)"].append(r2_score(training_y, pca_training_y_pred))
    continuous_features_r2["Validation Data (With PCA)"].append(r2_score(validation_y, pca_validation_y_pred))

    os.makedirs(f"{output_path}/Visualizations/With PCA/{feature_name}", exist_ok=True)

    for example_index, example_text in enumerate(visualization_examples):
        visualize_score_map(
            f"Visualizations/With PCA/{feature_name}/example{example_index}.png",
            example_text,
            tokenizer,
            lambda tokens: [ compute_continuous_feature[feature_name](token) for token in tokens ],
            lambda tokens: pca_regression.predict(pca.transform(scaler.transform(model.model.embed_tokens(torch.tensor(tokens))))),
            max_feature_scores[feature_name]
        )

logistic_features_r2 = {
    "Training Data (Without PCA)": [],
    "Validation Data (Without PCA)": [],
    "Training Data (With PCA)": [],
    "Validation Data (With PCA)": []
}

for feature_index, feature_name in enumerate(logistic_features):
    print(f"Evaluating {feature_name}")

    scaler = StandardScaler()

    training_sample_count = round(training_split * len(tokens))

    training_x = scaler.fit_transform(token_embeddings[:training_sample_count])
    validation_x = scaler.transform(token_embeddings[training_sample_count:])

    training_y = logistic_feature_scores[feature_index, :training_sample_count]
    validation_y = logistic_feature_scores[feature_index, training_sample_count:]

    regression = LogisticRegression(C=logistic_c)
    regression.fit(training_x, training_y)

    training_y_pred = regression.predict(training_x)
    validation_y_pred = regression.predict(validation_x)
    
    logistic_features_r2["Training Data (Without PCA)"].append(r2_score(training_y, training_y_pred))
    logistic_features_r2["Validation Data (Without PCA)"].append(r2_score(validation_y, validation_y_pred))

    os.makedirs(f"{output_path}/Visualizations/Without PCA/{feature_name}", exist_ok=True)

    for example_index, example_text in enumerate(visualization_examples):
        visualize_score_map(
            f"Visualizations/Without PCA/{feature_name}/example{example_index}.png",
            example_text,
            tokenizer,
            lambda tokens: [ compute_logistic_feature[feature_name](token) for token in tokens ],
            lambda tokens: regression.predict(scaler.transform(model.model.embed_tokens(torch.tensor(tokens)))),
            max_feature_scores[feature_name]
        )

    pca = PCA(n_components=pca_count)
    
    pca_training_x = pca.fit_transform(training_x)
    pca_validation_x = pca.transform(validation_x)

    pca_regression = LogisticRegression(C=pca_logistic_c)
    pca_regression.fit(pca_training_x, training_y)

    pca_training_y_pred = pca_regression.predict(pca_training_x)
    pca_validation_y_pred = pca_regression.predict(pca_validation_x)
    
    logistic_features_r2["Training Data (With PCA)"].append(r2_score(training_y, pca_training_y_pred))
    logistic_features_r2["Validation Data (With PCA)"].append(r2_score(validation_y, pca_validation_y_pred))

    os.makedirs(f"{output_path}/Visualizations/With PCA/{feature_name}", exist_ok=True)

    for example_index, example_text in enumerate(visualization_examples):
        visualize_score_map(
            f"Visualizations/With PCA/{feature_name}/example{example_index}.png",
            example_text,
            tokenizer,
            lambda tokens: [ compute_logistic_feature[feature_name](token) for token in tokens ],
            lambda tokens: pca_regression.predict(pca.transform(scaler.transform(model.model.embed_tokens(torch.tensor(tokens))))),
            max_feature_scores[feature_name]
        )

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

os.makedirs(output_path, exist_ok=True)

for feature_index, feature_name in enumerate(continuous_features):
    plot(
        f"{feature_name}.png",
        f"Predicting {feature_name}",
        [ evaluation_set for evaluation_set in continuous_features_r2.keys() ],
        [ r2_scores[feature_index] for r2_scores in continuous_features_r2.values() ]
    )

for feature_index, feature_name in enumerate(logistic_features):
    plot(
        f"{feature_name}.png",
        f"Predicting {feature_name}",
        [ evaluation_set for evaluation_set in logistic_features_r2.keys() ],
        [ r2_scores[feature_index] for r2_scores in logistic_features_r2.values() ]
    )

for evaluation_set in continuous_features_r2.keys():
    plot(
        f"{evaluation_set}.png",
        f"Predicting Features From {evaluation_set}",
        continuous_features + logistic_features,
        (
            [ continuous_features_r2[evaluation_set][feature_index] for feature_index in range(len(continuous_features)) ] +
            [ logistic_features_r2[evaluation_set][feature_index] for feature_index in range(len(logistic_features)) ]
        )
    )