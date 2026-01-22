from collections import Counter
import numpy as np
from scipy import sparse
from pygsp import utils
from . import fourier, difference  # prevent circular import in Python < 3.5
def get_edge_list(self):
    """Return an edge list, an alternative representation of the graph.

        The weighted adjacency matrix is the canonical form used in this
        package to represent a graph as it is the easiest to work with when
        considering spectral methods.

        Returns
        -------
        v_in : vector of int
        v_out : vector of int
        weights : vector of float

        Examples
        --------
        >>> G = graphs.Logo()
        >>> v_in, v_out, weights = G.get_edge_list()
        >>> v_in.shape, v_out.shape, weights.shape
        ((3131,), (3131,), (3131,))

        """
    if self.is_directed():
        raise NotImplementedError('Directed graphs not supported yet.')
    else:
        v_in, v_out = sparse.tril(self.W).nonzero()
        weights = self.W[v_in, v_out]
        weights = weights.toarray().squeeze()
        assert self.Ne == v_in.size == v_out.size == weights.size
        return (v_in, v_out, weights)