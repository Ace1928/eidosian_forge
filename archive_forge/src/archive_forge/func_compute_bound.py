import logging
import numpy as np
from scipy.special import digamma, gammaln
from scipy import optimize
from gensim import utils, matutils
from gensim.models import ldamodel
def compute_bound(self, sstats, totals):
    """Compute the maximized lower bound achieved for the log probability of the true posterior.

        Uses the formula presented in the appendix of the DTM paper (formula no. 5).

        Parameters
        ----------
        sstats : numpy.ndarray
            Sufficient statistics for a particular topic. Corresponds to matrix beta in the linked paper for the first
            time slice, expected shape (`self.vocab_len`, `num_topics`).
        totals : list of int of length `len(self.time_slice)`
            The totals for each time slice.

        Returns
        -------
        float
            The maximized lower bound.

        """
    w = self.vocab_len
    t = self.num_time_slices
    term_1 = 0
    term_2 = 0
    term_3 = 0
    val = 0
    ent = 0
    chain_variance = self.chain_variance
    self.mean, self.fwd_mean = (np.array(x) for x in zip(*(self.compute_post_mean(w, self.chain_variance) for w in range(w))))
    self.zeta = self.update_zeta()
    val = sum((self.variance[w][0] - self.variance[w][t] for w in range(w))) / 2 * chain_variance
    logger.info('Computing bound, all times')
    for t in range(1, t + 1):
        term_1 = 0.0
        term_2 = 0.0
        ent = 0.0
        for w in range(w):
            m = self.mean[w][t]
            prev_m = self.mean[w][t - 1]
            v = self.variance[w][t]
            term_1 += np.power(m - prev_m, 2) / (2 * chain_variance) - v / chain_variance - np.log(chain_variance)
            term_2 += sstats[w][t - 1] * m
            ent += np.log(v) / 2
        term_3 = -totals[t - 1] * np.log(self.zeta[t - 1])
        val += term_2 + term_3 + ent - term_1
    return val