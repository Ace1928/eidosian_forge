import itertools
import sys
import platform
import pytest
import numpy as np
from numpy.testing import (
def test_out_is_res(self):
    a = np.arange(9).reshape(3, 3)
    res = np.einsum('...ij,...jk->...ik', a, a, out=a)
    assert res is a