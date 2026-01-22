import numpy as np
from pygsp import utils
from . import approximations
def compute_frame(self, **kwargs):
    """Compute the associated frame.

        The size of the returned matrix operator :math:`D` is N x MN, where M
        is the number of filters and N the number of nodes. Multiplying this
        matrix with a set of signals is equivalent to analyzing them with the
        associated filterbank. Though computing this matrix is a rather
        inefficient way of doing it.

        The frame is defined as follows:

        .. math:: g_i(L) = U g_i(\\Lambda) U^*,

        where :math:`g` is the filter kernel, :math:`L` is the graph Laplacian,
        :math:`\\Lambda` is a diagonal matrix of the Laplacian's eigenvalues,
        and :math:`U` is the Fourier basis, i.e. its columns are the
        eigenvectors of the Laplacian.

        Parameters
        ----------
        kwargs: dict
            Parameters to be passed to the :meth:`analyze` method.

        Returns
        -------
        frame : ndarray
            Matrix of size N x MN.

        See also
        --------
        filter: more efficient way to filter signals

        Examples
        --------
        Filtering signals as a matrix multiplication.

        >>> G = graphs.Sensor(N=1000, seed=42)
        >>> G.estimate_lmax()
        >>> f = filters.MexicanHat(G, Nf=6)
        >>> s = np.random.uniform(size=G.N)
        >>>
        >>> frame = f.compute_frame()
        >>> frame.shape
        (1000, 1000, 6)
        >>> frame = frame.reshape(G.N, -1).T
        >>> s1 = np.dot(frame, s)
        >>> s1 = s1.reshape(G.N, -1)
        >>>
        >>> s2 = f.filter(s)
        >>> np.all((s1 - s2) < 1e-10)
        True

        """
    if self.G.N > 2000:
        _logger.warning('Creating a big matrix, you can use other means.')
    s = np.identity(self.G.N)
    return self.filter(s, **kwargs)