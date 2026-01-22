from math import sqrt
from numba import cuda, float32, int16, int32, int64, uint32, void
from numba.cuda import compile_ptx, compile_ptx_for_current_device
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
def test_c_abi_with_abi_name(self):
    abi_info = {'abi_name': '_Z4funcii'}
    ptx, resty = compile_ptx(f_module, int32(int32, int32), device=True, abi='c', abi_info=abi_info)
    self.assertRegex(ptx, '\\.visible\\s+\\.func\\s+\\(\\.param\\s+\\.b32\\s+func_retval0\\)\\s+_Z4funcii\\(')