import numpy as np
from numpy import array
from numpy.testing import (assert_equal, assert_,
import pytest
from pytest import raises as assert_raises
from scipy._lib._testutils import check_free_memory
from scipy._lib._util import check_random_state
from scipy.sparse import (csr_matrix, coo_matrix,
from scipy.sparse._construct import rand as sprand
def test_bmat_return_type(self):
    """This can be removed after sparse matrix is removed"""
    bmat = construct.bmat
    Fl, Gl = ([[1, 2], [3, 4]], [[7], [5]])
    Fm, Gm = (csr_matrix(Fl), csr_matrix(Gl))
    Fa, Ga = (csr_array(Fl), csr_array(Gl))
    assert isinstance(bmat([[Fa, Ga]], format='csr'), sparray)
    assert isinstance(bmat([[Fm, Gm]], format='csr'), spmatrix)
    assert isinstance(bmat([[None, Fa], [Ga, None]], format='csr'), sparray)
    assert isinstance(bmat([[None, Fm], [Ga, None]], format='csr'), sparray)
    assert isinstance(bmat([[None, Fm], [Gm, None]], format='csr'), spmatrix)
    assert isinstance(bmat([[None, Fl], [Gl, None]], format='csr'), spmatrix)
    assert isinstance(bmat([[Ga, Ga]], format='csr'), sparray)
    assert isinstance(bmat([[Gm, Ga]], format='csr'), sparray)
    assert isinstance(bmat([[Ga, Gm]], format='csr'), sparray)
    assert isinstance(bmat([[Gm, Gm]], format='csr'), spmatrix)
    assert isinstance(bmat([[Fa, Fm]], format='csr'), sparray)
    assert isinstance(bmat([[Fm, Fm]], format='csr'), spmatrix)
    assert isinstance(bmat([[Gm.tocsc(), Ga.tocsc()]], format='csc'), sparray)
    assert isinstance(bmat([[Gm.tocsc(), Gm.tocsc()]], format='csc'), spmatrix)
    assert isinstance(bmat([[Fa.tocsc(), Fm.tocsc()]], format='csr'), sparray)
    assert isinstance(bmat([[Fm.tocsc(), Fm.tocsc()]], format='csr'), spmatrix)
    assert isinstance(bmat([[Gl, Ga]], format='csr'), sparray)
    assert isinstance(bmat([[Gm.tocsc(), Ga]], format='csr'), sparray)
    assert isinstance(bmat([[Gm.tocsc(), Gm]], format='csr'), spmatrix)
    assert isinstance(bmat([[Gm, Gm]], format='csc'), spmatrix)