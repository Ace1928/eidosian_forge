import numpy as np
from pygsp import utils
from . import approximations
def estimate_frame_bounds(self, min=0, max=None, N=1000, use_eigenvalues=False):
    """Estimate lower and upper frame bounds.

        The frame bounds are estimated using the vector :code:`np.linspace(min,
        max, N)` with min=0 and max=G.lmax by default. The eigenvalues G.e can
        be used instead if you set use_eigenvalues to True.

        Parameters
        ----------
        min : float
            The lowest value the filter bank is evaluated at. By default
            filtering is bounded by the eigenvalues of G, i.e. min = 0.
        max : float
            The largest value the filter bank is evaluated at. By default
            filtering is bounded by the eigenvalues of G, i.e. max = G.lmax.
        N : int
            Number of points where the filter bank is evaluated.
            Default is 1000.
        use_eigenvalues : bool
            Set to True to use the Laplacian eigenvalues instead.

        Returns
        -------
        A : float
            Lower frame bound of the filter bank.
        B : float
            Upper frame bound of the filter bank.

        Examples
        --------
        >>> G = graphs.Logo()
        >>> G.compute_fourier_basis()
        >>> f = filters.MexicanHat(G)

        Bad estimation:

        >>> A, B = f.estimate_frame_bounds(min=1, max=20, N=100)
        >>> print('A={:.3f}, B={:.3f}'.format(A, B))
        A=0.125, B=0.270

        Better estimation:

        >>> A, B = f.estimate_frame_bounds()
        >>> print('A={:.3f}, B={:.3f}'.format(A, B))
        A=0.177, B=0.270

        Best estimation:

        >>> G.compute_fourier_basis()
        >>> A, B = f.estimate_frame_bounds(use_eigenvalues=True)
        >>> print('A={:.3f}, B={:.3f}'.format(A, B))
        A=0.177, B=0.270

        The Itersine filter bank defines a tight frame:

        >>> f = filters.Itersine(G)
        >>> A, B = f.estimate_frame_bounds(use_eigenvalues=True)
        >>> print('A={:.3f}, B={:.3f}'.format(A, B))
        A=1.000, B=1.000

        """
    if max is None:
        max = self.G.lmax
    if use_eigenvalues:
        x = self.G.e
    else:
        x = np.linspace(min, max, N)
    sum_filters = np.sum(np.abs(self.evaluate(x) ** 2), axis=0)
    return (sum_filters.min(), sum_filters.max())