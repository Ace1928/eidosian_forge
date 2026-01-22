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
def test_deprecation_dtype_attribute_is_dtype(self):

    class dt:
        dtype = 'f8'

    class vdt(np.void):
        dtype = 'f,f'
    self.assert_deprecated(lambda: np.dtype(dt))
    self.assert_deprecated(lambda: np.dtype(dt()))
    self.assert_deprecated(lambda: np.dtype(vdt))
    self.assert_deprecated(lambda: np.dtype(vdt(1)))