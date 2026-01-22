import collections
import hashlib
import json
import warnings
import numpy as np
from tensorflow.python.util.tf_export import keras_export
def sequences_to_texts_generator(self, sequences):
    """Transforms each sequence in `sequences` to a list of texts(strings).

        Each sequence has to a list of integers.
        In other words, sequences should be a list of sequences

        Only top `num_words-1` most frequent words will be taken into account.
        Only words known by the tokenizer will be taken into account.

        Args:
            sequences: A list of sequences.

        Yields:
            Yields individual texts.
        """
    num_words = self.num_words
    oov_token_index = self.word_index.get(self.oov_token)
    for seq in sequences:
        vect = []
        for num in seq:
            word = self.index_word.get(num)
            if word is not None:
                if num_words and num >= num_words:
                    if oov_token_index is not None:
                        vect.append(self.index_word[oov_token_index])
                else:
                    vect.append(word)
            elif self.oov_token is not None:
                vect.append(self.index_word[oov_token_index])
        vect = ' '.join(vect)
        yield vect