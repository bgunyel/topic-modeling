from sentence_transformers import SentenceTransformer
import umap
import hdbscan

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import utils
import constants


def cluster_docs(list_docs):
    sentence_transformer = SentenceTransformer('distilbert-base-nli-mean-tokens')
    embeddings = sentence_transformer.encode(list_docs, show_progress_bar=True)

    umap_embeddings = umap.UMAP(n_neighbors=15,
                                n_components=10,
                                metric='cosine',
                                random_state=constants.RANDOM_SEED).fit_transform(embeddings)
    cluster = hdbscan.HDBSCAN(min_cluster_size=15,
                              metric='euclidean',
                              cluster_selection_method='eom').fit(umap_embeddings)
    labels = cluster.labels_
    return embeddings, labels


def visualize_clusters(embeddings, labels):

    umap_data = umap.UMAP(n_neighbors=15,
                          n_components=2,
                          min_dist=0.0,
                          metric='cosine',
                          random_state=constants.RANDOM_SEED).fit_transform(embeddings)
    result = pd.DataFrame(umap_data, columns=['x', 'y'])
    result[constants.LABEL] = labels

    clustered = result.loc[result[constants.LABEL] != -1, :]

    plt.figure(figsize=(18, 8))
    sns.scatterplot(data=clustered, x='x', y='y', hue=constants.LABEL, palette='husl')
    plt.show()
