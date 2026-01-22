import numpy as np
import scipy.sparse as sp
from multiprocessing.pool import ThreadPool
from functools import partial
from . import utils
from . import tokenizers
from parlai.utils.logging import logger
def batch_closest_docs(self, queries, k=1, num_workers=None):
    """
        Process a batch of closest_docs requests multithreaded.

        Note: we can use plain threads here as scipy is outside of the GIL.
        """
    with ThreadPool(num_workers) as threads:
        closest_docs = partial(self.closest_docs, k=k)
        results = threads.map(closest_docs, queries)
    return results