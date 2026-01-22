import time
from datetime import date
import numpy as np
from numpy.testing import (
from numpy.lib._iotools import (
def test_keep_default_zero(self):
    """Check that we don't lose a default of 0"""
    converter = StringConverter(int, default=0, missing_values='N/A')
    assert_equal(converter.default, 0)