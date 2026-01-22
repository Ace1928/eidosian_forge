import warnings
from llvmlite import ir
from numba.cuda.cudadrv import nvvm, runtime
from numba.cuda.testing import unittest
from numba.cuda.cudadrv.nvvm import LibDevice, NvvmError, NVVM
from numba.cuda.testing import skip_on_cudasim
def test_nvvm_warning(self):
    m = ir.Module('test_nvvm_warning')
    m.triple = 'nvptx64-nvidia-cuda'
    m.data_layout = NVVM().data_layout
    nvvm.add_ir_version(m)
    fty = ir.FunctionType(ir.VoidType(), [])
    kernel = ir.Function(m, fty, name='inlinekernel')
    builder = ir.IRBuilder(kernel.append_basic_block('entry'))
    builder.ret_void()
    nvvm.set_cuda_kernel(kernel)
    kernel.attributes.add('noinline')
    with warnings.catch_warnings(record=True) as w:
        nvvm.llvm_to_ptx(str(m))
    self.assertEqual(len(w), 1)
    self.assertIn('overriding noinline attribute', str(w[0]))