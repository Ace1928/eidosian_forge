import numpy as np
from pygsp import utils
def gft(self, s):
    """Compute the graph Fourier transform.

        The graph Fourier transform of a signal :math:`s` is defined as

        .. math:: \\hat{s} = U^* s,

        where :math:`U` is the Fourier basis attr:`U` and :math:`U^*` denotes
        the conjugate transpose or Hermitian transpose of :math:`U`.

        Parameters
        ----------
        s : ndarray
            Graph signal in the vertex domain.

        Returns
        -------
        s_hat : ndarray
            Representation of s in the Fourier domain.

        Examples
        --------
        >>> G = graphs.Logo()
        >>> G.compute_fourier_basis()
        >>> s = np.random.normal(size=(G.N, 5, 1))
        >>> s_hat = G.gft(s)
        >>> s_star = G.igft(s_hat)
        >>> np.all((s - s_star) < 1e-10)
        True

        """
    if s.shape[0] != self.N:
        raise ValueError('First dimension should be the number of nodes G.N = {}, got {}.'.format(self.N, s.shape))
    U = np.conjugate(self.U)
    return np.tensordot(U, s, ([0], [0]))