# Unembedding Token Rarity

Uncovering connections between LLM token representations and semantic rarity.

![Predicting Inverse Collection Frequency from Word Embeddings of Collection Terms](experiments/2/results/Validation%20Data%20(Without%20PCA).png)

Classical IR features like inverse frequency are powerful indicators of rarity, and help ranking schemes like BM25 weight term importance. While neural reranking models often produce similar results to classical methods, it is unclear what features these model uses and how they are represented in the network.

To better understand how an important feature like term rarity is captured, we try to predict classical per-token rarity metrics as a linear function of dense token embeddings. A high R^2 value indicates the token embedding is highly predictive, showing the model has learned to embed signals related to semantic rarity.

Establishing how LLMs capture rarity may eventually help us interpret more complicated feature circuits and simplify model structure.

## Rarity Metrics in the Top 500 Principle Components

![Predicting Rarity Features From Input Embeddings of Document Terms](experiments/2/results/Validation%20Data%20(With%20PCA).png)

## Visualizing Log Inverse Document Frequency Predicted without PCA

Red represents the actual inverse document frequency of a token.

Blue represents the predicted inverse collection frequency of a token.

![Rarity Heat Map for Example Sentence 1](experiments/2/results/Visualizations/Without%20PCA/Log%20Inverse%20Frequency/example0.png)

![Rarity Heat Map for Example Sentence 2](experiments/2/results/Visualizations/Without%20PCA/Log%20Inverse%20Frequency/example1.png)

![Rarity Heat Map for Example Sentence 3](experiments/2/results/Visualizations/Without%20PCA/Log%20Inverse%20Frequency/example2.png)

## Visualizing Stopwords Predicted with PCA

Red represents the actual stopword tokens.

Blue represents the predicted stopword tokens.

![Rarity Heat Map for Example Sentence 1](experiments/2/results/Visualizations/With%20PCA/Is%20Stopword/example0.png)

![Rarity Heat Map for Example Sentence 1](experiments/2/results/Visualizations/With%20PCA/Is%20Stopword/example1.png)

![Rarity Heat Map for Example Sentence 1](experiments/2/results/Visualizations/With%20PCA/Is%20Stopword/example2.png)