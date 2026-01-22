import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_complex_item(self):
    self._assert_func(complex(1, 2), complex(1, 2))
    self._assert_func(complex(1, np.nan), complex(1, np.nan))
    self._assert_func(complex(np.inf, np.nan), complex(np.inf, np.nan))
    self._test_not_equal(complex(1, np.nan), complex(1, 2))
    self._test_not_equal(complex(np.nan, 1), complex(1, np.nan))
    self._test_not_equal(complex(np.nan, np.inf), complex(np.nan, 2))