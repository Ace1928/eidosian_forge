import numpy as np
import scipy.sparse as sp
from multiprocessing.pool import ThreadPool
from functools import partial
from . import utils
from . import tokenizers
from parlai.utils.logging import logger
def closest_docs(self, query, k=1, matrix=None):
    """
        Closest docs by dot product between query and documents in tfidf weighted word
        vector space.

        matrix arg can be provided to be used instead of internal doc matrix.
        """
    spvec = self.text2spvec(query)
    res = spvec * matrix if matrix is not None else spvec * self.doc_mat
    if len(res.data) <= k:
        o_sort = np.argsort(-res.data)
    else:
        o = np.argpartition(-res.data, k)[0:k]
        o_sort = o[np.argsort(-res.data[o])]
    doc_scores = res.data[o_sort]
    doc_ids = res.indices[o_sort]
    return (doc_ids, doc_scores)