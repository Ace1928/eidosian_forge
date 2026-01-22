import pytest
import numpy as np
import numpy.testing as npt
import scipy.sparse
import scipy.sparse.linalg as spla
from scipy._lib._util import VisibleDeprecationWarning
@pytest.mark.parametrize(('fmt', 'fn'), (('bsr', scipy.sparse.isspmatrix_bsr), ('coo', scipy.sparse.isspmatrix_coo), ('csc', scipy.sparse.isspmatrix_csc), ('csr', scipy.sparse.isspmatrix_csr), ('dia', scipy.sparse.isspmatrix_dia), ('dok', scipy.sparse.isspmatrix_dok), ('lil', scipy.sparse.isspmatrix_lil)))
def test_isspmatrix_format(fmt, fn):
    m = scipy.sparse.eye(3, format=fmt)
    a = scipy.sparse.csr_array(m).asformat(fmt)
    assert not isinstance(m, scipy.sparse.sparray)
    assert isinstance(a, scipy.sparse.sparray)
    assert not fn(a)
    assert fn(m)
    assert not fn(a.todense())
    assert not fn(m.todense())