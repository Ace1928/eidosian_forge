import logging
import sys
import itertools
import warnings
from numbers import Integral
from typing import Iterable
from numpy import (
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from gensim.corpora.dictionary import Dictionary
from gensim.utils import deprecated
def most_similar_cosmul(self, positive=None, negative=None, topn=10, restrict_vocab=None):
    """Find the top-N most similar words, using the multiplicative combination objective,
        proposed by `Omer Levy and Yoav Goldberg "Linguistic Regularities in Sparse and Explicit Word Representations"
        <http://www.aclweb.org/anthology/W14-1618>`_. Positive words still contribute positively towards the similarity,
        negative words negatively, but with less susceptibility to one large distance dominating the calculation.
        In the common analogy-solving case, of two positive and one negative examples,
        this method is equivalent to the "3CosMul" objective (equation (4)) of Levy and Goldberg.

        Additional positive or negative examples contribute to the numerator or denominator,
        respectively - a potentially sensible but untested extension of the method.
        With a single positive example, rankings will be the same as in the default
        :meth:`~gensim.models.keyedvectors.KeyedVectors.most_similar`.

        Allows calls like most_similar_cosmul('dog', 'cat'), as a shorthand for
        most_similar_cosmul(['dog'], ['cat']) where 'dog' is positive and 'cat' negative

        Parameters
        ----------
        positive : list of str, optional
            List of words that contribute positively.
        negative : list of str, optional
            List of words that contribute negatively.
        topn : int or None, optional
            Number of top-N similar words to return, when `topn` is int. When `topn` is None,
            then similarities for all words are returned.
        restrict_vocab : int or None, optional
            Optional integer which limits the range of vectors which are searched for most-similar values.
            For example, restrict_vocab=10000 would only check the first 10000 node vectors in the vocabulary order.
            This may be meaningful if vocabulary is sorted by descending frequency.


        Returns
        -------
        list of (str, float) or numpy.array
            When `topn` is int, a sequence of (word, similarity) is returned.
            When `topn` is None, then similarities for all words are returned as a
            one-dimensional numpy array with the size of the vocabulary.

        """
    if isinstance(topn, Integral) and topn < 1:
        return []
    positive = _ensure_list(positive)
    negative = _ensure_list(negative)
    self.init_sims()
    if isinstance(positive, str):
        positive = [positive]
    if isinstance(negative, str):
        negative = [negative]
    all_words = {self.get_index(word) for word in positive + negative if not isinstance(word, ndarray) and word in self.key_to_index}
    positive = [self.get_vector(word, norm=True) if isinstance(word, str) else word for word in positive]
    negative = [self.get_vector(word, norm=True) if isinstance(word, str) else word for word in negative]
    if not positive:
        raise ValueError('cannot compute similarity with no input')
    pos_dists = [(1 + dot(self.vectors, term) / self.norms) / 2 for term in positive]
    neg_dists = [(1 + dot(self.vectors, term) / self.norms) / 2 for term in negative]
    dists = prod(pos_dists, axis=0) / (prod(neg_dists, axis=0) + 1e-06)
    if not topn:
        return dists
    best = matutils.argsort(dists, topn=topn + len(all_words), reverse=True)
    result = [(self.index_to_key[sim], float(dists[sim])) for sim in best if sim not in all_words]
    return result[:topn]