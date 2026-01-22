import itertools
import logging
import multiprocessing as mp
import sys
from collections import Counter
import numpy as np
import scipy.sparse as sps
from gensim import utils
from gensim.models.word2vec import Word2Vec
def merge_accumulators(self, accumulators):
    """Merge the list of accumulators into a single `WordOccurrenceAccumulator` with all
        occurrence and co-occurrence counts, and a `num_docs` that reflects the total observed
        by all the individual accumulators.

        """
    accumulator = WordOccurrenceAccumulator(self.relevant_ids, self.dictionary)
    for other_accumulator in accumulators:
        accumulator.merge(other_accumulator)
    accumulator._symmetrize()
    logger.info('accumulated word occurrence stats for %d virtual documents', accumulator.num_docs)
    return accumulator