import re
import sys
import pytest
import numpy as np
import numpy.core._multiarray_tests as mt
from numpy.testing import assert_warns, IS_PYPY
def test_too_large(self):
    with pytest.raises(ValueError):
        self.conv(2 ** 64)