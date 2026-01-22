import collections
import hashlib
import json
import warnings
import numpy as np
from tensorflow.python.util.tf_export import keras_export
def fit_on_sequences(self, sequences):
    """Updates internal vocabulary based on a list of sequences.

        Required before using `sequences_to_matrix`
        (if `fit_on_texts` was never called).

        Args:
            sequences: A list of sequence.
                A "sequence" is a list of integer word indices.
        """
    self.document_count += len(sequences)
    for seq in sequences:
        seq = set(seq)
        for i in seq:
            self.index_docs[i] += 1