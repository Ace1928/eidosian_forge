import logging
import numpy as np
from numpy import ones, vstack, float32 as REAL
import gensim.models._fasttext_bin
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors, prep_vectors
from gensim import utils
from gensim.utils import deprecated
from gensim.models import keyedvectors  # noqa: E402
def recalc_char_ngram_buckets(self):
    """
        Scan the vocabulary, calculate ngrams and their hashes, and cache the list of ngrams for each known word.

        """
    if self.bucket == 0:
        self.buckets_word = [np.array([], dtype=np.uint32)] * len(self.index_to_key)
        return
    self.buckets_word = [None] * len(self.index_to_key)
    for i, word in enumerate(self.index_to_key):
        self.buckets_word[i] = np.array(ft_ngram_hashes(word, self.min_n, self.max_n, self.bucket), dtype=np.uint32)