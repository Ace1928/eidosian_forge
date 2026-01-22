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
def wmdistance(self, document1, document2, norm=True):
    """Compute the Word Mover's Distance between two documents.

        When using this code, please consider citing the following papers:

        * `Rémi Flamary et al. "POT: Python Optimal Transport"
          <https://jmlr.org/papers/v22/20-451.html>`_
        * `Matt Kusner et al. "From Word Embeddings To Document Distances"
          <http://proceedings.mlr.press/v37/kusnerb15.pdf>`_.

        Parameters
        ----------
        document1 : list of str
            Input document.
        document2 : list of str
            Input document.
        norm : boolean
            Normalize all word vectors to unit length before computing the distance?
            Defaults to True.

        Returns
        -------
        float
            Word Mover's distance between `document1` and `document2`.

        Warnings
        --------
        This method only works if `POT <https://pypi.org/project/POT/>`_ is installed.

        If one of the documents have no words that exist in the vocab, `float('inf')` (i.e. infinity)
        will be returned.

        Raises
        ------
        ImportError
            If `POT <https://pypi.org/project/POT/>`_  isn't installed.

        """
    from ot import emd2
    len_pre_oov1 = len(document1)
    len_pre_oov2 = len(document2)
    document1 = [token for token in document1 if token in self]
    document2 = [token for token in document2 if token in self]
    diff1 = len_pre_oov1 - len(document1)
    diff2 = len_pre_oov2 - len(document2)
    if diff1 > 0 or diff2 > 0:
        logger.info('Removed %d and %d OOV words from document 1 and 2 (respectively).', diff1, diff2)
    if not document1 or not document2:
        logger.warning('At least one of the documents had no words that were in the vocabulary.')
        return float('inf')
    dictionary = Dictionary(documents=[document1, document2])
    vocab_len = len(dictionary)
    if vocab_len == 1:
        return 0.0
    doclist1 = list(set(document1))
    doclist2 = list(set(document2))
    v1 = np.array([self.get_vector(token, norm=norm) for token in doclist1])
    v2 = np.array([self.get_vector(token, norm=norm) for token in doclist2])
    doc1_indices = dictionary.doc2idx(doclist1)
    doc2_indices = dictionary.doc2idx(doclist2)
    distance_matrix = zeros((vocab_len, vocab_len), dtype=double)
    distance_matrix[np.ix_(doc1_indices, doc2_indices)] = cdist(v1, v2)
    if abs(np_sum(distance_matrix)) < 1e-08:
        logger.info('The distance matrix is all zeros. Aborting (returning inf).')
        return float('inf')

    def nbow(document):
        d = zeros(vocab_len, dtype=double)
        nbow = dictionary.doc2bow(document)
        doc_len = len(document)
        for idx, freq in nbow:
            d[idx] = freq / float(doc_len)
        return d
    d1 = nbow(document1)
    d2 = nbow(document2)
    return emd2(d1, d2, distance_matrix)