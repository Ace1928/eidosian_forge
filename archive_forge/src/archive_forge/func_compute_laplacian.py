from collections import Counter
import numpy as np
from scipy import sparse
from pygsp import utils
from . import fourier, difference  # prevent circular import in Python < 3.5
def compute_laplacian(self, lap_type='combinatorial'):
    """Compute a graph Laplacian.

        The result is accessible by the L attribute.

        Parameters
        ----------
        lap_type : 'combinatorial', 'normalized'
            The type of Laplacian to compute. Default is combinatorial.

        Notes
        -----
        For undirected graphs, the combinatorial Laplacian is defined as

        .. math:: L = D - W,

        where :math:`W` is the weight matrix and :math:`D` the degree matrix,
        and the normalized Laplacian is defined as

        .. math:: L = I - D^{-1/2} W D^{-1/2},

        where :math:`I` is the identity matrix.

        Examples
        --------
        >>> G = graphs.Sensor(50)
        >>> G.L.shape
        (50, 50)
        >>>
        >>> G.compute_laplacian('combinatorial')
        >>> G.compute_fourier_basis()
        >>> -1e-10 < G.e[0] < 1e-10  # Smallest eigenvalue close to 0.
        True
        >>>
        >>> G.compute_laplacian('normalized')
        >>> G.compute_fourier_basis(recompute=True)
        >>> -1e-10 < G.e[0] < 1e-10 < G.e[-1] < 2  # Spectrum in [0, 2].
        True

        """
    if lap_type not in ['combinatorial', 'normalized']:
        raise ValueError('Unknown Laplacian type {}'.format(lap_type))
    self.lap_type = lap_type
    if self.is_directed():
        if lap_type == 'combinatorial':
            D1 = sparse.diags(np.ravel(self.W.sum(0)), 0)
            D2 = sparse.diags(np.ravel(self.W.sum(1)), 0)
            self.L = 0.5 * (D1 + D2 - self.W - self.W.T).tocsc()
        elif lap_type == 'normalized':
            raise NotImplementedError('Directed graphs with normalized Laplacian not supported yet.')
    elif lap_type == 'combinatorial':
        D = sparse.diags(np.ravel(self.W.sum(1)), 0)
        self.L = (D - self.W).tocsc()
    elif lap_type == 'normalized':
        d = np.power(self.dw, -0.5)
        D = sparse.diags(np.ravel(d), 0).tocsc()
        self.L = sparse.identity(self.N) - D * self.W * D