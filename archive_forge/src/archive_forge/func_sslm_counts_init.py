import logging
import numpy as np
from scipy.special import digamma, gammaln
from scipy import optimize
from gensim import utils, matutils
from gensim.models import ldamodel
def sslm_counts_init(self, obs_variance, chain_variance, sstats):
    """Initialize the State Space Language Model with LDA sufficient statistics.

        Called for each topic-chain and initializes initial mean, variance and Topic-Word probabilities
        for the first time-slice.

        Parameters
        ----------
        obs_variance : float, optional
            Observed variance used to approximate the true and forward variance.
        chain_variance : float
            Gaussian parameter defined in the beta distribution to dictate how the beta values evolve over time.
        sstats : numpy.ndarray
            Sufficient statistics of the LDA model. Corresponds to matrix beta in the linked paper for time slice 0,
            expected shape (`self.vocab_len`, `num_topics`).

        """
    W = self.vocab_len
    T = self.num_time_slices
    log_norm_counts = np.copy(sstats)
    log_norm_counts /= sum(log_norm_counts)
    log_norm_counts += 1.0 / W
    log_norm_counts /= sum(log_norm_counts)
    log_norm_counts = np.log(log_norm_counts)
    self.obs = np.repeat(log_norm_counts, T, axis=0).reshape(W, T)
    self.obs_variance = obs_variance
    self.chain_variance = chain_variance
    for w in range(W):
        self.variance[w], self.fwd_variance[w] = self.compute_post_variance(w, self.chain_variance)
        self.mean[w], self.fwd_mean[w] = self.compute_post_mean(w, self.chain_variance)
    self.zeta = self.update_zeta()
    self.e_log_prob = self.compute_expected_log_prob()