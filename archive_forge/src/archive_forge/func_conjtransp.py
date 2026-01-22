import itertools
import numpy as np
from ._matrix import spmatrix
from ._base import _spbase, sparray, issparse
from ._index import IndexMixin
from ._sputils import (isdense, getdtype, isshape, isintlike, isscalarlike,
def conjtransp(self):
    """Return the conjugate transpose."""
    M, N = self.shape
    new = self._dok_container((N, M), dtype=self.dtype)
    new._dict.update((((right, left), np.conj(val)) for (left, right), val in self.items()))
    return new