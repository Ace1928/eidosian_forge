import logging
import numpy as np
from scipy.special import digamma, gammaln
from scipy import optimize
from gensim import utils, matutils
from gensim.models import ldamodel
def update_obs(self, sstats, totals):
    """Optimize the bound with respect to the observed variables.

        TODO:
        This is by far the slowest function in the whole algorithm.
        Replacing or improving the performance of this would greatly speed things up.

        Parameters
        ----------
        sstats : numpy.ndarray
            Sufficient statistics for a particular topic. Corresponds to matrix beta in the linked paper for the first
            time slice, expected shape (`self.vocab_len`, `num_topics`).
        totals : list of int of length `len(self.time_slice)`
            The totals for each time slice.

        Returns
        -------
        (numpy.ndarray of float, numpy.ndarray of float)
            The updated optimized values for obs and the zeta variational parameter.

        """
    OBS_NORM_CUTOFF = 2
    STEP_SIZE = 0.01
    TOL = 0.001
    W = self.vocab_len
    T = self.num_time_slices
    runs = 0
    mean_deriv_mtx = np.zeros((T, T + 1))
    norm_cutoff_obs = None
    for w in range(W):
        w_counts = sstats[w]
        counts_norm = 0
        for i in range(len(w_counts)):
            counts_norm += w_counts[i] * w_counts[i]
        counts_norm = np.sqrt(counts_norm)
        if counts_norm < OBS_NORM_CUTOFF and norm_cutoff_obs is not None:
            obs = self.obs[w]
            norm_cutoff_obs = np.copy(obs)
        else:
            if counts_norm < OBS_NORM_CUTOFF:
                w_counts = np.zeros(len(w_counts))
            for t in range(T):
                mean_deriv_mtx[t] = self.compute_mean_deriv(w, t, mean_deriv_mtx[t])
            deriv = np.zeros(T)
            args = (self, w_counts, totals, mean_deriv_mtx, w, deriv)
            obs = self.obs[w]
            model = 'DTM'
            if model == 'DTM':
                obs = optimize.fmin_cg(f=f_obs, fprime=df_obs, x0=obs, gtol=TOL, args=args, epsilon=STEP_SIZE, disp=0)
            if model == 'DIM':
                pass
            runs += 1
            if counts_norm < OBS_NORM_CUTOFF:
                norm_cutoff_obs = obs
            self.obs[w] = obs
    self.zeta = self.update_zeta()
    return (self.obs, self.zeta)