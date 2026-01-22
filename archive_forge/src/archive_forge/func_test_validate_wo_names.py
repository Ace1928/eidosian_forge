import time
from datetime import date
import numpy as np
from numpy.testing import (
from numpy.lib._iotools import (
def test_validate_wo_names(self):
    """Test validate no names"""
    namelist = None
    validator = NameValidator()
    assert_(validator(namelist) is None)
    assert_equal(validator(namelist, nbfields=3), ['f0', 'f1', 'f2'])