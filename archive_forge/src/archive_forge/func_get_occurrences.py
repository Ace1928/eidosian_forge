import itertools
import logging
import multiprocessing as mp
import sys
from collections import Counter
import numpy as np
import scipy.sparse as sps
from gensim import utils
from gensim.models.word2vec import Word2Vec
def get_occurrences(self, word):
    """Return number of docs the word occurs in, once `accumulate` has been called."""
    try:
        self.token2id[word]
    except KeyError:
        word = self.dictionary.id2token[word]
    return self.model.get_vecattr(word, 'count')