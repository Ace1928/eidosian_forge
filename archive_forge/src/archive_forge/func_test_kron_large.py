import numpy as np
from numpy import array
from numpy.testing import (assert_equal, assert_,
import pytest
from pytest import raises as assert_raises
from scipy._lib._testutils import check_free_memory
from scipy._lib._util import check_random_state
from scipy.sparse import (csr_matrix, coo_matrix,
from scipy.sparse._construct import rand as sprand
def test_kron_large(self):
    n = 2 ** 16
    a = construct.diags_array([1], shape=(1, n), offsets=n - 1)
    b = construct.diags_array([1], shape=(n, 1), offsets=1 - n)
    construct.kron(a, a)
    construct.kron(b, b)