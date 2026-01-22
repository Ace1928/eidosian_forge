import collections
import hashlib
import json
import warnings
import numpy as np
from tensorflow.python.util.tf_export import keras_export
def sequences_to_matrix(self, sequences, mode='binary'):
    """Converts a list of sequences into a Numpy matrix.

        Args:
            sequences: list of sequences
                (a sequence is a list of integer word indices).
            mode: one of "binary", "count", "tfidf", "freq"

        Returns:
            A Numpy matrix.

        Raises:
            ValueError: In case of invalid `mode` argument,
                or if the Tokenizer requires to be fit to sample data.
        """
    if not self.num_words:
        if self.word_index:
            num_words = len(self.word_index) + 1
        else:
            raise ValueError('Specify a dimension (`num_words` argument), or fit on some text data first.')
    else:
        num_words = self.num_words
    if mode == 'tfidf' and (not self.document_count):
        raise ValueError('Fit the Tokenizer on some data before using tfidf mode.')
    x = np.zeros((len(sequences), num_words))
    for i, seq in enumerate(sequences):
        if not seq:
            continue
        counts = collections.defaultdict(int)
        for j in seq:
            if j >= num_words:
                continue
            counts[j] += 1
        for j, c in list(counts.items()):
            if mode == 'count':
                x[i][j] = c
            elif mode == 'freq':
                x[i][j] = c / len(seq)
            elif mode == 'binary':
                x[i][j] = 1
            elif mode == 'tfidf':
                tf = 1 + np.log(c)
                idf = np.log(1 + self.document_count / (1 + self.index_docs.get(j, 0)))
                x[i][j] = tf * idf
            else:
                raise ValueError('Unknown vectorization mode:', mode)
    return x