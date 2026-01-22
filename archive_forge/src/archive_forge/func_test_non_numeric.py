import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_non_numeric(self):
    self._assert_func('ab', 'ab')
    self._test_not_equal('ab', 'abb')