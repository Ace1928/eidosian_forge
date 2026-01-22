import collections
import hashlib
import json
import warnings
import numpy as np
from tensorflow.python.util.tf_export import keras_export
def texts_to_sequences(self, texts):
    """Transforms each text in texts to a sequence of integers.

        Only top `num_words-1` most frequent words will be taken into account.
        Only words known by the tokenizer will be taken into account.

        Args:
            texts: A list of texts (strings).

        Returns:
            A list of sequences.
        """
    return list(self.texts_to_sequences_generator(texts))