from numpy.testing import (assert_, assert_array_equal)
import numpy as np
import pytest
from numpy.random import Generator, MT19937
def test_beta_ridiculously_small_parameters(self):
    tiny = np.finfo(1.0).tiny
    x = self.mt19937.beta(tiny / 32, tiny / 40, size=50)
    assert not np.any(np.isnan(x))