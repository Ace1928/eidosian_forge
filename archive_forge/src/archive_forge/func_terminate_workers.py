import itertools
import logging
import multiprocessing as mp
import sys
from collections import Counter
import numpy as np
import scipy.sparse as sps
from gensim import utils
from gensim.models.word2vec import Word2Vec
def terminate_workers(self, input_q, output_q, workers, interrupted=False):
    """Wait until all workers have transmitted their WordOccurrenceAccumulator instances, then terminate each.

        Warnings
        --------
        We do not use join here because it has been shown to have some issues
        in Python 2.7 (and even in later versions). This method also closes both the input and output queue.
        If `interrupted` is False (normal execution), a None value is placed on the input queue for
        each worker. The workers are looking for this sentinel value and interpret it as a signal to
        terminate themselves. If `interrupted` is True, a KeyboardInterrupt occurred. The workers are
        programmed to recover from this and continue on to transmit their results before terminating.
        So in this instance, the sentinel values are not queued, but the rest of the execution
        continues as usual.

        """
    if not interrupted:
        for _ in workers:
            input_q.put(None, block=True)
    accumulators = []
    while len(accumulators) != len(workers):
        accumulators.append(output_q.get())
    logger.info('%d accumulators retrieved from output queue', len(accumulators))
    for worker in workers:
        if worker.is_alive():
            worker.terminate()
    input_q.close()
    output_q.close()
    return accumulators