import warnings
from llvmlite import ir
from numba.cuda.cudadrv import nvvm, runtime
from numba.cuda.testing import unittest
from numba.cuda.cudadrv.nvvm import LibDevice, NvvmError, NVVM
from numba.cuda.testing import skip_on_cudasim
def test_libdevice_load(self):
    libdevice = LibDevice()
    self.assertEqual(libdevice.bc[:4], b'BC\xc0\xde')