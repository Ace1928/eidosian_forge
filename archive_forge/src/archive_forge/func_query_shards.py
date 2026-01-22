import logging
import itertools
import os
import heapq
import warnings
import numpy
import scipy.sparse
from gensim import interfaces, utils, matutils
def query_shards(self, query):
    """Apply shard[query] to each shard in `self.shards`. Used internally.

        Parameters
        ----------
        query : {iterable of list of (int, number) , list of (int, number))}
            Document in BoW format or corpus of documents.

        Returns
        -------
        (None, list of individual shard query results)
            Query results.

        """
    args = zip([query] * len(self.shards), self.shards)
    if PARALLEL_SHARDS and PARALLEL_SHARDS > 1:
        logger.debug('spawning %i query processes', PARALLEL_SHARDS)
        pool = multiprocessing.Pool(PARALLEL_SHARDS)
        result = pool.imap(query_shard, args, chunksize=1 + len(self.shards) / PARALLEL_SHARDS)
    else:
        pool = None
        result = map(query_shard, args)
    return (pool, result)