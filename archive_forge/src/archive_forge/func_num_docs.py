import itertools
import logging
import multiprocessing as mp
import sys
from collections import Counter
import numpy as np
import scipy.sparse as sps
from gensim import utils
from gensim.models.word2vec import Word2Vec
@num_docs.setter
def num_docs(self, num):
    self._num_docs = num
    if self._num_docs % self.log_every == 0:
        logger.info('%s accumulated stats from %d documents', self.__class__.__name__, self._num_docs)