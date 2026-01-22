import time
from datetime import date
import numpy as np
from numpy.testing import (
from numpy.lib._iotools import (
def test_keep_missing_values(self):
    """Check that we're not losing missing values"""
    converter = StringConverter(int, default=0, missing_values='N/A')
    assert_equal(converter.missing_values, {'', 'N/A'})