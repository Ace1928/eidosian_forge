import pytest
import numpy as np
from numpy.testing import (
from numpy.lib.index_tricks import (
def test_low_dim_handling(self):
    a = np.zeros(3, int)
    with assert_raises_regex(ValueError, 'at least 2-d'):
        fill_diagonal(a, 5)