import logging
import numpy as np
from scipy.special import digamma, gammaln
from scipy import optimize
from gensim import utils, matutils
from gensim.models import ldamodel
def fit_sslm(self, sstats):
    """Fits variational distribution.

        This is essentially the m-step.
        Maximizes the approximation of the true posterior for a particular topic using the provided sufficient
        statistics. Updates the values using :meth:`~gensim.models.ldaseqmodel.sslm.update_obs` and
        :meth:`~gensim.models.ldaseqmodel.sslm.compute_expected_log_prob`.

        Parameters
        ----------
        sstats : numpy.ndarray
            Sufficient statistics for a particular topic. Corresponds to matrix beta in the linked paper for the
            current time slice, expected shape (`self.vocab_len`, `num_topics`).

        Returns
        -------
        float
            The lower bound for the true posterior achieved using the fitted approximate distribution.

        """
    W = self.vocab_len
    bound = 0
    old_bound = 0
    sslm_fit_threshold = 1e-06
    sslm_max_iter = 2
    converged = sslm_fit_threshold + 1
    self.variance, self.fwd_variance = (np.array(x) for x in zip(*(self.compute_post_variance(w, self.chain_variance) for w in range(W))))
    totals = sstats.sum(axis=0)
    iter_ = 0
    model = 'DTM'
    if model == 'DTM':
        bound = self.compute_bound(sstats, totals)
    if model == 'DIM':
        bound = self.compute_bound_fixed(sstats, totals)
    logger.info('initial sslm bound is %f', bound)
    while converged > sslm_fit_threshold and iter_ < sslm_max_iter:
        iter_ += 1
        old_bound = bound
        self.obs, self.zeta = self.update_obs(sstats, totals)
        if model == 'DTM':
            bound = self.compute_bound(sstats, totals)
        if model == 'DIM':
            bound = self.compute_bound_fixed(sstats, totals)
        converged = np.fabs((bound - old_bound) / old_bound)
        logger.info('iteration %i iteration lda seq bound is %f convergence is %f', iter_, bound, converged)
    self.e_log_prob = self.compute_expected_log_prob()
    return bound