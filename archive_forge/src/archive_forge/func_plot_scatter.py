from abc import ABC, abstractmethod
import numpy as np
from scipy import stats
from statsmodels.graphics import utils
def plot_scatter(self, sample=None, nobs=500, random_state=None, ax=None):
    """Sample the copula and plot.

        Parameters
        ----------
        sample : array-like, optional
            The sample to plot.  If not provided (the default), a sample
            is generated.
        nobs : int, optional
            Number of samples to generate from the copula.
        random_state : {None, int, numpy.random.Generator}, optional
            If `seed` is None then the legacy singleton NumPy generator.
            This will change after 0.13 to use a fresh NumPy ``Generator``,
            so you should explicitly pass a seeded ``Generator`` if you
            need reproducible results.
            If `seed` is an int, a new ``Generator`` instance is used,
            seeded with `seed`.
            If `seed` is already a ``Generator`` instance then that instance is
            used.
        ax : AxesSubplot, optional
            If given, this subplot is used to plot in instead of a new figure
            being created.

        Returns
        -------
        fig : Figure
            If `ax` is None, the created figure.  Otherwise the figure to which
            `ax` is connected.
        sample : array_like (n, d)
            Sample from the copula.

        See Also
        --------
        statsmodels.tools.rng_qrng.check_random_state
        """
    if self.k_dim != 2:
        raise ValueError('Can only plot 2-dimensional Copula.')
    if sample is None:
        sample = self.rvs(nobs=nobs, random_state=random_state)
    fig, ax = utils.create_mpl_ax(ax)
    ax.scatter(sample[:, 0], sample[:, 1])
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    return (fig, sample)