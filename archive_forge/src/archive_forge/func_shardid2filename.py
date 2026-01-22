import logging
import itertools
import os
import heapq
import warnings
import numpy
import scipy.sparse
from gensim import interfaces, utils, matutils
def shardid2filename(self, shardid):
    """Get shard file by `shardid`.

        Parameters
        ----------
        shardid : int
            Shard index.

        Return
        ------
        str
            Path to shard file.

        """
    if self.output_prefix.endswith('.'):
        return '%s%s' % (self.output_prefix, shardid)
    else:
        return '%s.%s' % (self.output_prefix, shardid)