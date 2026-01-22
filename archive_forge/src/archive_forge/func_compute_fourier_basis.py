import numpy as np
from pygsp import utils
def compute_fourier_basis(self, recompute=False):
    """Compute the Fourier basis of the graph (cached).

        The result is cached and accessible by the :attr:`U`, :attr:`e`,
        :attr:`lmax`, and :attr:`mu` properties.

        Parameters
        ----------
        recompute: bool
            Force to recompute the Fourier basis if already existing.

        Notes
        -----
        'G.compute_fourier_basis()' computes a full eigendecomposition of
        the graph Laplacian :math:`L` such that:

        .. math:: L = U \\Lambda U^*,

        where :math:`\\Lambda` is a diagonal matrix of eigenvalues and the
        columns of :math:`U` are the eigenvectors.

        *G.e* is a vector of length *G.N* containing the Laplacian
        eigenvalues. The largest eigenvalue is stored in *G.lmax*.
        The eigenvectors are stored as column vectors of *G.U* in the same
        order that the eigenvalues. Finally, the coherence of the
        Fourier basis is found in *G.mu*.

        References
        ----------
        See :cite:`chung1997spectral`.

        Examples
        --------
        >>> G = graphs.Torus()
        >>> G.compute_fourier_basis()
        >>> G.U.shape
        (256, 256)
        >>> G.e.shape
        (256,)
        >>> G.lmax == G.e[-1]
        True
        >>> G.mu < 1
        True

        """
    if hasattr(self, '_e') and hasattr(self, '_U') and (not recompute):
        return
    assert self.L.shape == (self.N, self.N)
    if self.N > 3000:
        self.logger.warning('Computing the full eigendecomposition of a large matrix ({0} x {0}) may take some time.'.format(self.N))
    self._e, self._U = np.linalg.eigh(self.L.toarray())
    assert -1e-12 < self._e[0] < 1e-12
    self._e[0] = 0
    if self.lap_type == 'normalized':
        assert self._e[-1] <= 2
    assert np.max(self._e) == self._e[-1]
    self._lmax = self._e[-1]
    self._mu = np.max(np.abs(self._U))