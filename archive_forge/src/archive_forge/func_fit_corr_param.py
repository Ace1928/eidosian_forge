from abc import ABC, abstractmethod
import numpy as np
from scipy import stats
from statsmodels.graphics import utils
def fit_corr_param(self, data):
    """Copula correlation parameter using Kendall's tau of sample data.

        Parameters
        ----------
        data : array_like
            Sample data used to fit `theta` using Kendall's tau.

        Returns
        -------
        corr_param : float
            Correlation parameter of the copula, ``theta`` in Archimedean and
            pearson correlation in elliptical.
            If k_dim > 2, then average tau is used.
        """
    x = np.asarray(data)
    if x.shape[1] == 2:
        tau = stats.kendalltau(x[:, 0], x[:, 1])[0]
    else:
        k = self.k_dim
        taus = [stats.kendalltau(x[..., i], x[..., j])[0] for i in range(k) for j in range(i + 1, k)]
        tau = np.mean(taus)
    return self._arg_from_tau(tau)