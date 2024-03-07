# Fine-Tuning Transformers for Toponym Resolution
## A Contextual Embedding Approach to Candidate Ranking
---
## Abstract
We introduce a new approach to toponym resolution, leveraging transformer-based Siamese networks to disambiguate geographical references in unstructured text. Our methodology consists of two steps: the generation of location candidates using the GeoNames gazetteer, and the ranking of these candidates based on their semantic similarity to the toponym in its document context. The core of the proposed method lies in the adaption of SentenceTransformer models, originally designed for sentence similarity tasks, to toponym resolution by fine-tuning them on geographically annotated English news article datasets (Local Global Lexicon, GeoWebNews, and TR-News). The models are used to generate contextual embeddings of both toponyms and textual representations of location candidates, which are then used to rank candidates using cosine similarity. The results suggest that the fine-tuned models outperform existing solutions in several key metrics.

## Demonstration
For a practical demonstration of our approach, we have prepared the `demo.ipynb` notebook. It provides a step-by-step overview of the entire process, from data preparation and pre-processing to model fine-tuning and evaluation.