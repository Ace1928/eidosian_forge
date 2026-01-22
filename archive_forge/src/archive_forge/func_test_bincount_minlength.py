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
def test_bincount_minlength(self):
    self.assert_deprecated(lambda: np.bincount([1, 2, 3], minlength=None))