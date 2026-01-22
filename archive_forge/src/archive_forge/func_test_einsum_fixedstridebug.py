import itertools
import sys
import platform
import pytest
import numpy as np
from numpy.testing import (
def test_einsum_fixedstridebug(self):
    A = np.arange(2 * 3).reshape(2, 3).astype(np.float32)
    B = np.arange(2 * 3 * 2731).reshape(2, 3, 2731).astype(np.int16)
    es = np.einsum('cl, cpx->lpx', A, B)
    tp = np.tensordot(A, B, axes=(0, 0))
    assert_equal(es, tp)
    A = np.arange(3 * 3).reshape(3, 3).astype(np.float64)
    B = np.arange(3 * 3 * 64 * 64).reshape(3, 3, 64, 64).astype(np.float32)
    es = np.einsum('cl, cpxy->lpxy', A, B)
    tp = np.tensordot(A, B, axes=(0, 0))
    assert_equal(es, tp)