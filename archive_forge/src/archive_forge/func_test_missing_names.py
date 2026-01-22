import time
from datetime import date
import numpy as np
from numpy.testing import (
from numpy.lib._iotools import (
def test_missing_names(self):
    """Test validate missing names"""
    namelist = ('a', 'b', 'c')
    validator = NameValidator()
    assert_equal(validator(namelist), ['a', 'b', 'c'])
    namelist = ('', 'b', 'c')
    assert_equal(validator(namelist), ['f0', 'b', 'c'])
    namelist = ('a', 'b', '')
    assert_equal(validator(namelist), ['a', 'b', 'f0'])
    namelist = ('', 'f0', '')
    assert_equal(validator(namelist), ['f1', 'f0', 'f2'])