import csv
import logging
from numbers import Integral
import sys
import time
from collections import defaultdict, Counter
import numpy as np
from numpy import random as np_random, float32 as REAL
from scipy.stats import spearmanr
from gensim import utils, matutils
from gensim.models.keyedvectors import KeyedVectors
def score_function(self, embedding, trie, term_1, term_2):
    """Compute predicted score - extent to which `term_1` is a type of `term_2`.

        Parameters
        ----------
        embedding : :class:`~gensim.models.poincare.PoincareKeyedVectors`
            Embedding to use for computing predicted score.
        trie : :class:`pygtrie.Trie`
            Trie to use for finding matching vocab terms for input terms.
        term_1 : str
            Input term.
        term_2 : str
            Input term.

        Returns
        -------
        float
            Predicted score (the extent to which `term_1` is a type of `term_2`).

        """
    try:
        word_1_terms = self.find_matching_terms(trie, term_1)
        word_2_terms = self.find_matching_terms(trie, term_2)
    except KeyError:
        raise ValueError('No matching terms found for either %s or %s' % (term_1, term_2))
    min_distance = np.inf
    min_term_1, min_term_2 = (None, None)
    for term_1 in word_1_terms:
        for term_2 in word_2_terms:
            distance = embedding.distance(term_1, term_2)
            if distance < min_distance:
                min_term_1, min_term_2 = (term_1, term_2)
                min_distance = distance
    assert min_term_1 is not None and min_term_2 is not None
    vector_1, vector_2 = (embedding.get_vector(min_term_1), embedding.get_vector(min_term_2))
    norm_1, norm_2 = (np.linalg.norm(vector_1), np.linalg.norm(vector_2))
    return -1 * (1 + self.alpha * (norm_2 - norm_1)) * min_distance