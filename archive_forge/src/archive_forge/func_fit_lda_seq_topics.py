import logging
import numpy as np
from scipy.special import digamma, gammaln
from scipy import optimize
from gensim import utils, matutils
from gensim.models import ldamodel
def fit_lda_seq_topics(self, topic_suffstats):
    """Fit the sequential model topic-wise.

        Parameters
        ----------
        topic_suffstats : numpy.ndarray
            Sufficient statistics of the current model, expected shape (`self.vocab_len`, `num_topics`).

        Returns
        -------
        float
            The sum of the optimized lower bounds for all topics.

        """
    lhood = 0
    for k, chain in enumerate(self.topic_chains):
        logger.info('Fitting topic number %i', k)
        lhood_term = sslm.fit_sslm(chain, topic_suffstats[k])
        lhood += lhood_term
    return lhood