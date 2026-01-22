import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_negative_zero(self):
    self._test_not_equal(np.PZERO, np.NZERO)