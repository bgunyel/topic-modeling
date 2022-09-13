import torch
from sentence_transformers import SentenceTransformer
import umap
import hdbscan

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import utils
import constants
import topic_model


def main(name):
    print(name)

    df_train, df_test = utils.read_train_test_data()
    train_data = [str(t) for t in df_train[constants.EXCERPT].values]

    embeddings, labels = topic_model.cluster_docs(train_data)
    topic_model.visualize_clusters(embeddings=embeddings, labels=labels)





    dummy = -32


if __name__ == '__main__':

    if not torch.cuda.is_available():
        raise Exception('CUDA IS NOT INITIALIZED!')

    main('Topic Modeling')
