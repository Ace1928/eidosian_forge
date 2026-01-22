from numpy.testing import assert_equal
from scipy.sparse import csr_matrix, csr_array, sparray
import numpy as np
from scipy.sparse import _extract
def test_array_vs_matrix(self):
    for A in self.cases:
        assert isinstance(_extract.tril(A), sparray)
        assert isinstance(_extract.triu(A), sparray)
        M = csr_matrix(A)
        assert not isinstance(_extract.tril(M), sparray)
        assert not isinstance(_extract.triu(M), sparray)