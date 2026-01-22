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
def test_filter_ignore_numba_deprecation_only(self):
    with warnings.catch_warnings():
        warnings.simplefilter('error', category=DeprecationWarning)
        warnings.simplefilter('error', category=PendingDeprecationWarning)
        warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
        warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
        with self.assertRaises(DeprecationWarning):
            warnings.warn(DeprecationWarning('this is not ignored'))
        with self.assertRaises(PendingDeprecationWarning):
            warnings.warn(PendingDeprecationWarning('this is not ignored'))
        warnings.warn(NumbaDeprecationWarning('this is ignored'))
        warnings.warn(NumbaPendingDeprecationWarning('this is ignored'))
        warnings.simplefilter('error', category=NumbaDeprecationWarning)
        warnings.simplefilter('error', category=NumbaPendingDeprecationWarning)
        with self.assertRaises(DeprecationWarning):
            warnings.warn(NumbaDeprecationWarning('this is not ignored'))
        with self.assertRaises(PendingDeprecationWarning):
            warnings.warn(NumbaPendingDeprecationWarning('this is not ignored'))