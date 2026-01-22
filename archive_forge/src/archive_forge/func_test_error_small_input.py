import pytest
import numpy as np
from numpy.testing import (
from numpy.lib.index_tricks import (
def test_error_small_input(self):
    x = np.ones(7)
    with assert_raises_regex(ValueError, 'at least 2-d'):
        diag_indices_from(x)