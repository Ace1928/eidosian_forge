import os
import sys
import subprocess
import threading
from numba import cuda
from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,
from numba.tests.support import captured_stdout
@skip_on_cudasim('Simulator does not hit device library search code path')
@unittest.skipIf(not sys.platform.startswith('linux'), 'linux only')
def test_cuda_find_lib_errors(self):
    """
        This tests that the find_libs works as expected in the case of an
        environment variable being used to set the path.
        """
    locs = ['lib', 'lib64']
    looking_for = None
    for l in locs:
        looking_for = os.path.join(os.path.sep, l)
        if os.path.exists(looking_for):
            break
    if looking_for is not None:
        out, err = self.run_test_in_separate_process('NUMBA_CUDA_DRIVER', looking_for)
        self.assertTrue(out is not None)
        self.assertTrue(err is not None)