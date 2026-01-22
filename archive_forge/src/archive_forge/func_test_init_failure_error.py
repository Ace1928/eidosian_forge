import multiprocessing as mp
import os
from numba import cuda
from numba.cuda.cudadrv.driver import CudaAPIError, driver
from numba.cuda.cudadrv.error import CudaSupportError
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
def test_init_failure_error(self):
    expected = 'CUDA_ERROR_UNKNOWN (999)'
    self._test_init_failure(initialization_error_test, expected)