import numpy as np
import warnings
from numba.cuda.testing import unittest
from numba.cuda.testing import (skip_on_cudasim, skip_if_cuda_includes_missing)
from numba.cuda.testing import CUDATestCase, test_data_dir
from numba.cuda.cudadrv.driver import (CudaAPIError, Linker,
from numba.cuda.cudadrv.error import NvrtcError
from numba.cuda import require_context
from numba.tests.support import ignore_internal_warnings
from numba import cuda, void, float64, int64, int32, typeof, float32
def test_linking_cu_error(self):
    bar = cuda.declare_device('bar', 'int32(int32)')
    link = str(test_data_dir / 'error.cu')
    with self.assertRaises(NvrtcError) as e:

        @cuda.jit('void(int32)', link=[link])
        def kernel(x):
            bar(x)
    msg = e.exception.args[0]
    self.assertIn('NVRTC Compilation failure', msg)
    self.assertIn('identifier "SYNTAX" is undefined', msg)
    self.assertIn('in the compilation of "error.cu"', msg)