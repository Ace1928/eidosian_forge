import pytest
import numpy as np
from numpy.testing import assert_allclose
from pytest import raises as assert_raises
from scipy import sparse
from scipy.sparse import csgraph
from scipy._lib._util import np_long, np_ulong
def check_int_type(mat):
    return np.issubdtype(mat.dtype, np.signedinteger) or np.issubdtype(mat.dtype, np_ulong)