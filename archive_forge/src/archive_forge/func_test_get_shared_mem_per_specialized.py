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
def test_get_shared_mem_per_specialized(self):
    compiled = cuda.jit(simple_smem)
    compiled_specialized = compiled.specialize(np.zeros(100, dtype=np.int32), np.float64)
    shared_mem_size = compiled_specialized.get_shared_mem_per_block()
    self.assertEqual(shared_mem_size, 800)