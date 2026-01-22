import logging
import itertools
import os
import heapq
import warnings
import numpy
import scipy.sparse
from gensim import interfaces, utils, matutils
def get_similarities(self, query):
    """Get similarity between `query` and this index.

        Warnings
        --------
        Do not use this function directly; use the `self[query]` syntax instead.

        Parameters
        ----------
        query : {list of (int, number), iterable of list of (int, number), :class:`scipy.sparse.csr_matrix`}
            Document or collection of documents.

        Return
        ------
        :class:`numpy.ndarray`
            Similarity matrix (if maintain_sparsity=False) **OR**
        :class:`scipy.sparse.csc`
            otherwise

        """
    is_corpus, query = utils.is_corpus(query)
    if is_corpus:
        query = matutils.corpus2csc(query, self.index.shape[1], dtype=self.index.dtype)
    elif scipy.sparse.issparse(query):
        query = query.T
    elif isinstance(query, numpy.ndarray):
        if query.ndim == 1:
            query.shape = (1, len(query))
        query = scipy.sparse.csr_matrix(query, dtype=self.index.dtype).T
    else:
        query = matutils.corpus2csc([query], self.index.shape[1], dtype=self.index.dtype)
    result = self.index * query.tocsc()
    if result.shape[1] == 1 and (not is_corpus):
        result = result.toarray().flatten()
    elif self.maintain_sparsity:
        result = result.T
    else:
        result = result.toarray().T
    return result