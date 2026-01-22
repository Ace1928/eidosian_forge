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
def test_get_shared_mem_per_block(self):
    sig = void(int32[::1], typeof(np.int32))
    compiled = cuda.jit(sig)(simple_smem)
    shared_mem_size = compiled.get_shared_mem_per_block()
    self.assertEqual(shared_mem_size, 400)