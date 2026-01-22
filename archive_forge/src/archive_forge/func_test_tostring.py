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
def test_tostring(self):
    arr = np.array(list(b'test\xff'), dtype=np.uint8)
    self.assert_deprecated(arr.tostring)