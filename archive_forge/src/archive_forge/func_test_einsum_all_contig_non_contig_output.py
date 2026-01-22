import itertools
import sys
import platform
import pytest
import numpy as np
from numpy.testing import (
def test_einsum_all_contig_non_contig_output(self):
    x = np.ones((5, 5))
    out = np.ones(10)[::2]
    correct_base = np.ones(10)
    correct_base[::2] = 5
    np.einsum('mi,mi,mi->m', x, x, x, out=out)
    assert_array_equal(out.base, correct_base)
    out = np.ones(10)[::2]
    np.einsum('im,im,im->m', x, x, x, out=out)
    assert_array_equal(out.base, correct_base)
    out = np.ones((2, 2, 2))[..., 0]
    correct_base = np.ones((2, 2, 2))
    correct_base[..., 0] = 2
    x = np.ones((2, 2), np.float32)
    np.einsum('ij,jk->ik', x, x, out=out)
    assert_array_equal(out.base, correct_base)