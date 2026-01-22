import numpy as np
from pygsp import utils
def gft_windowed_gabor(self, s, k):
    """Gabor windowed graph Fourier transform.

        Parameters
        ----------
        s : ndarray
            Graph signal in the vertex domain.
        k : function
            Gabor kernel. See :class:`pygsp.filters.Gabor`.

        Returns
        -------
        s : ndarray
            Vertex-frequency representation of the signals.

        Examples
        --------
        >>> G = graphs.Logo()
        >>> s = np.random.normal(size=(G.N, 2))
        >>> s = G.gft_windowed_gabor(s, lambda x: x/(1.-x))
        >>> s.shape
        (1130, 2, 1130)

        """
    from pygsp import filters
    return filters.Gabor(self, k).filter(s)