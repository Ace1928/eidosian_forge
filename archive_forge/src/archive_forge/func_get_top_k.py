import collections
import itertools
import random
import string
import time
import numpy as np
import tensorflow.compat.v2 as tf
import keras.src as keras
from keras.src.layers.preprocessing import index_lookup
def get_top_k(dataset, k):
    """Python implementation of vocabulary building using a defaultdict."""
    counts = collections.defaultdict(int)
    for tensor in dataset:
        data = tensor.numpy()
        for element in data:
            counts[element] += 1
    sorted_vocab = [k for k, _ in sorted(counts.items(), key=lambda item: item[1], reverse=True)]
    if len(sorted_vocab) > k:
        sorted_vocab = sorted_vocab[:k]
    return sorted_vocab