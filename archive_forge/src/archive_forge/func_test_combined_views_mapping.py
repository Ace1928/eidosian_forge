import itertools
import sys
import platform
import pytest
import numpy as np
from numpy.testing import (
def test_combined_views_mapping(self):
    a = np.arange(9).reshape(1, 1, 3, 1, 3)
    b = np.einsum('bbcdc->d', a)
    assert_equal(b, [12])