import re
import numpy as np
from numpy.testing import assert_
def test_short_version():
    if np.version.release:
        assert_(np.__version__ == np.version.short_version, 'short_version mismatch in release version')
    else:
        assert_(np.__version__.split('+')[0] == np.version.short_version, 'short_version mismatch in development version')