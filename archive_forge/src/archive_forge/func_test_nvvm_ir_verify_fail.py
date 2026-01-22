import warnings
from llvmlite import ir
from numba.cuda.cudadrv import nvvm, runtime
from numba.cuda.testing import unittest
from numba.cuda.cudadrv.nvvm import LibDevice, NvvmError, NVVM
from numba.cuda.testing import skip_on_cudasim
def test_nvvm_ir_verify_fail(self):
    m = ir.Module('test_bad_ir')
    m.triple = 'unknown-unknown-unknown'
    m.data_layout = NVVM().data_layout
    nvvm.add_ir_version(m)
    with self.assertRaisesRegex(NvvmError, 'Invalid target triple'):
        nvvm.llvm_to_ptx(str(m))