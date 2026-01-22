import numpy as np
from numba import guvectorize, cuda
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
def test_gufunc_scalar_output(self):

    @guvectorize(['void(int32[:], int32[:])'], '(n)->()', target='cuda')
    def sum_row(inp, out):
        tmp = 0.0
        for i in range(inp.shape[0]):
            tmp += inp[i]
        out[0] = tmp
    inp = np.arange(300, dtype=np.int32).reshape(100, 3)
    out1 = np.empty(100, dtype=inp.dtype)
    out2 = np.empty(100, dtype=inp.dtype)
    dev_inp = cuda.to_device(inp)
    dev_out1 = cuda.to_device(out1, copy=False)
    sum_row(dev_inp, out=dev_out1)
    dev_out2 = sum_row(dev_inp)
    dev_out1.copy_to_host(out1)
    dev_out2.copy_to_host(out2)
    for i in range(inp.shape[0]):
        self.assertTrue(out1[i] == inp[i].sum())
        self.assertTrue(out2[i] == inp[i].sum())