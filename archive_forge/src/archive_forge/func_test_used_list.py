import warnings
from llvmlite import ir
from numba.cuda.cudadrv import nvvm, runtime
from numba.cuda.testing import unittest
from numba.cuda.cudadrv.nvvm import LibDevice, NvvmError, NVVM
from numba.cuda.testing import skip_on_cudasim
def test_used_list(self):
    m = ir.Module('test_used_list')
    m.triple = 'nvptx64-nvidia-cuda'
    m.data_layout = NVVM().data_layout
    nvvm.add_ir_version(m)
    fty = ir.FunctionType(ir.VoidType(), [ir.IntType(32)])
    kernel = ir.Function(m, fty, name='mycudakernel')
    bldr = ir.IRBuilder(kernel.append_basic_block('entry'))
    bldr.ret_void()
    nvvm.set_cuda_kernel(kernel)
    used_lines = [line for line in str(m).splitlines() if 'llvm.used' in line]
    msg = 'Expected exactly one @"llvm.used" array'
    self.assertEqual(len(used_lines), 1, msg)
    used_line = used_lines[0]
    self.assertIn('mycudakernel', used_line)
    self.assertIn('appending global', used_line)
    self.assertIn('section "llvm.metadata"', used_line)