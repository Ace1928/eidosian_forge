import os
import numpy as np
from numpy.testing import (
def test_include_dirs(self):
    include_dirs = [np.get_include()]
    for path in include_dirs:
        assert_(isinstance(path, str))
        assert_(path != '')