import logging
from itertools import chain
from copy import deepcopy
from shutil import copyfile
from os.path import isfile
from os import remove
import numpy as np  # for arrays, array broadcasting etc.
from scipy.special import gammaln  # gamma function utils
from gensim import utils
from gensim.models import LdaModel
from gensim.models.ldamodel import LdaState
from gensim.matutils import dirichlet_expectation, mean_absolute_difference
from gensim.corpora import MmCorpus
def log_perplexity(self, chunk, chunk_doc_idx=None, total_docs=None):
    """Calculate per-word likelihood bound, using the `chunk` of documents as evaluation corpus.

        Parameters
        ----------
        chunk : iterable of list of (int, float)
            Corpus in BoW format.
        chunk_doc_idx : numpy.ndarray, optional
            Assigns the value for document index.
        total_docs : int, optional
            Initializes the value for total number of documents.

        Returns
        -------
        float
            Value of per-word likelihood bound.

        """
    if total_docs is None:
        total_docs = len(chunk)
    corpus_words = sum((cnt for document in chunk for _, cnt in document))
    subsample_ratio = 1.0 * total_docs / len(chunk)
    perwordbound = self.bound(chunk, chunk_doc_idx, subsample_ratio=subsample_ratio) / (subsample_ratio * corpus_words)
    logger.info('%.3f per-word bound, %.1f perplexity estimate based on a corpus of %i documents with %i words', perwordbound, np.exp2(-perwordbound), len(chunk), corpus_words)
    return perwordbound