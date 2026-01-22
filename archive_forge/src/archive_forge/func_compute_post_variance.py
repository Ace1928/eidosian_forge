import logging
import numpy as np
from scipy.special import digamma, gammaln
from scipy import optimize
from gensim import utils, matutils
from gensim.models import ldamodel
def compute_post_variance(self, word, chain_variance):
    """Get the variance, based on the
        `Variational Kalman Filtering approach for Approximate Inference (section 3.1)
        <https://mimno.infosci.cornell.edu/info6150/readings/dynamic_topic_models.pdf>`_.

        This function accepts the word to compute variance for, along with the associated sslm class object,
        and returns the `variance` and the posterior approximation `fwd_variance`.

        Notes
        -----
        This function essentially computes Var[\\beta_{t,w}] for t = 1:T

        .. :math::

            fwd\\_variance[t] \\equiv E((beta_{t,w}-mean_{t,w})^2 |beta_{t}\\ for\\ 1:t) =
            (obs\\_variance / fwd\\_variance[t - 1] + chain\\_variance + obs\\_variance ) *
            (fwd\\_variance[t - 1] + obs\\_variance)

        .. :math::

            variance[t] \\equiv E((beta_{t,w}-mean\\_cap_{t,w})^2 |beta\\_cap_{t}\\ for\\ 1:t) =
            fwd\\_variance[t - 1] + (fwd\\_variance[t - 1] / fwd\\_variance[t - 1] + obs\\_variance)^2 *
            (variance[t - 1] - (fwd\\_variance[t-1] + obs\\_variance))

        Parameters
        ----------
        word: int
            The word's ID.
        chain_variance : float
            Gaussian parameter defined in the beta distribution to dictate how the beta values evolve over time.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            The first returned value is the variance of each word in each time slice, the second value is the
            inferred posterior variance for the same pairs.

        """
    INIT_VARIANCE_CONST = 1000
    T = self.num_time_slices
    variance = self.variance[word]
    fwd_variance = self.fwd_variance[word]
    fwd_variance[0] = chain_variance * INIT_VARIANCE_CONST
    for t in range(1, T + 1):
        if self.obs_variance:
            c = self.obs_variance / (fwd_variance[t - 1] + chain_variance + self.obs_variance)
        else:
            c = 0
        fwd_variance[t] = c * (fwd_variance[t - 1] + chain_variance)
    variance[T] = fwd_variance[T]
    for t in range(T - 1, -1, -1):
        if fwd_variance[t] > 0.0:
            c = np.power(fwd_variance[t] / (fwd_variance[t] + chain_variance), 2)
        else:
            c = 0
        variance[t] = c * (variance[t + 1] - chain_variance) + (1 - c) * fwd_variance[t]
    return (variance, fwd_variance)