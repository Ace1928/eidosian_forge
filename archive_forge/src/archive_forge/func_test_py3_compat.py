import os
import numpy as np
from numpy.testing import (
def test_py3_compat(self):

    class C:
        """Old-style class in python2, normal class in python3"""
        pass
    out = open(os.devnull, 'w')
    try:
        np.info(C(), output=out)
    except AttributeError:
        raise AssertionError()
    finally:
        out.close()