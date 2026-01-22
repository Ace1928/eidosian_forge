import multiprocessing
import os
from numba.core import config
from numba.cuda.cudadrv.runtime import runtime
from numba.cuda.testing import unittest, SerialMixin, skip_on_cudasim
from unittest.mock import patch
def test_is_supported_version_true(self):
    for v in SUPPORTED_VERSIONS:
        with patch.object(runtime, 'get_version', return_value=v):
            self.assertTrue(runtime.is_supported_version())