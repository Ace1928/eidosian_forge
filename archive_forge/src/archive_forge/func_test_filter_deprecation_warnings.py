import os
import subprocess
import sys
import warnings
import numpy as np
import unittest
from numba import jit
from numba.core.errors import (
from numba.core import errors
from numba.tests.support import ignore_internal_warnings
def test_filter_deprecation_warnings(self):
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        warnings.simplefilter('ignore', category=DeprecationWarning)
        warnings.simplefilter('ignore', category=PendingDeprecationWarning)
        warnings.warn(DeprecationWarning('this is ignored'))
        warnings.warn(PendingDeprecationWarning('this is ignored'))
        warnings.warn(NumbaDeprecationWarning('this is ignored'))
        warnings.warn(NumbaPendingDeprecationWarning('this is ignored'))
        with self.assertRaises(NumbaWarning):
            warnings.warn(NumbaWarning('this is not ignored'))