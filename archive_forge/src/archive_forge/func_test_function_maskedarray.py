import pytest
import numpy as np
from numpy.testing import assert_warns
from numpy.ma.testutils import assert_equal
from numpy.ma.core import MaskedArrayFutureWarning
import io
import textwrap
def test_function_maskedarray(self):
    return self._test_base(np.ma.argsort, np.ma.MaskedArray)