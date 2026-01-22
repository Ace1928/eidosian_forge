import logging
import itertools
import os
import heapq
import warnings
import numpy
import scipy.sparse
from gensim import interfaces, utils, matutils
def close_shard(self):
    """Force the latest shard to close (be converted to a matrix and stored to disk).
         Do nothing if no new documents added since last call.

        Notes
        -----
        The shard is closed even if it is not full yet (its size is smaller than `self.shardsize`).
        If documents are added later via :meth:`~gensim.similarities.docsim.MatrixSimilarity.add_documents`
        this incomplete shard will be loaded again and completed.

        """
    if not self.fresh_docs:
        return
    shardid = len(self.shards)
    issparse = 0.3 > 1.0 * self.fresh_nnz / (len(self.fresh_docs) * self.num_features)
    if issparse:
        index = SparseMatrixSimilarity(self.fresh_docs, num_terms=self.num_features, num_docs=len(self.fresh_docs), num_nnz=self.fresh_nnz)
    else:
        index = MatrixSimilarity(self.fresh_docs, num_features=self.num_features)
    logger.info('creating %s shard #%s', 'sparse' if issparse else 'dense', shardid)
    shard = Shard(self.shardid2filename(shardid), index)
    shard.num_best = self.num_best
    shard.num_nnz = self.fresh_nnz
    self.shards.append(shard)
    self.fresh_docs, self.fresh_nnz = ([], 0)