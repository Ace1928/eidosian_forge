import copy
import logging
import itertools
import decimal
from functools import cache
import numpy
from ._vertex import (VertexCacheField, VertexCacheIndex)
def deg_simplex(self, S, proj=None):
    """Test a simplex S for degeneracy (linear dependence in R^dim).

        Parameters
        ----------
        S : np.array
            Simplex with rows as vertex vectors
        proj : array, optional,
            If the projection S[1:] - S[0] is already
            computed it can be added as an optional argument.
        """
    if proj is None:
        proj = S[1:] - S[0]
    if numpy.linalg.det(proj) == 0.0:
        return True
    else:
        return False