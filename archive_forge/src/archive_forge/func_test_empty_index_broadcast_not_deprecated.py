import datetime
import operator
import warnings
import pytest
import tempfile
import re
import sys
import numpy as np
from numpy.testing import (
from numpy.core._multiarray_tests import fromstring_null_term_c_api
def test_empty_index_broadcast_not_deprecated(self):
    arr = np.ones((2, 2, 2))
    index = ([[3], [2]], [])
    self.assert_not_deprecated(arr.__getitem__, args=(index,))
    self.assert_not_deprecated(arr.__setitem__, args=(index, np.empty((2, 0, 2))))