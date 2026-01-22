import itertools
import logging
import multiprocessing as mp
import sys
from collections import Counter
import numpy as np
import scipy.sparse as sps
from gensim import utils
from gensim.models.word2vec import Word2Vec
def partial_accumulate(self, texts, window_size):
    """Meant to be called several times to accumulate partial results.

        Notes
        -----
        The final accumulation should be performed with the `accumulate` method as opposed to this one.
        This method does not ensure the co-occurrence matrix is in lil format and does not
        symmetrize it after accumulation.

        """
    self._current_doc_num = -1
    self._token_at_edge = None
    self._counter.clear()
    super(WordOccurrenceAccumulator, self).accumulate(texts, window_size)
    for combo, count in self._counter.items():
        self._co_occurrences[combo] += count
    return self