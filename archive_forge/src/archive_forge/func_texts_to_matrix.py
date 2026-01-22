import collections
import hashlib
import json
import warnings
import numpy as np
from tensorflow.python.util.tf_export import keras_export
def texts_to_matrix(self, texts, mode='binary'):
    """Convert a list of texts to a Numpy matrix.

        Args:
            texts: list of strings.
            mode: one of "binary", "count", "tfidf", "freq".

        Returns:
            A Numpy matrix.
        """
    sequences = self.texts_to_sequences(texts)
    return self.sequences_to_matrix(sequences, mode=mode)