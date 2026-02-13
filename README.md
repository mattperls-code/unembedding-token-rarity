# Unembedding Document Frequency

Uncovering connections between LLM token representations and semantic rarity.

![Predicting Inverse Collection Frequency from Word Embeddings of Collection Terms](results/Collection%20Terms/Inverse%20Collection%20Frequency/Validation.png)

Classical IR features like inverse frequency are powerful indicators of rarity, and help ranking schemes like BM25 weight term importance. While neural reranking models often produce similar results to classical methods, it is unclear what features these model uses and how they are represented in the network.

To better understand how an important feature like term rarity is captured, we try to predict classical per-token rarity metrics as a linear function of dense token embeddings. A high R^2 value indicates the token embedding is highly predictive, showing the model has learned to embed signals related to semantic rarity.

Establishing how LLMs capture rarity may eventually help us interpret more complicated feature circuits and simplify model structure.