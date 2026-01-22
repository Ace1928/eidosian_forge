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
def test_linking_cu(self):
    bar = cuda.declare_device('bar', 'int32(int32)')
    link = str(test_data_dir / 'jitlink.cu')

    @cuda.jit(link=[link])
    def kernel(r, x):
        i = cuda.grid(1)
        if i < len(r):
            r[i] = bar(x[i])
    x = np.arange(10, dtype=np.int32)
    r = np.zeros_like(x)
    kernel[1, 32](r, x)
    expected = x * 2
    np.testing.assert_array_equal(r, expected)