import numpy as np
from numba import from_dtype, cuda
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
def test_record_alignment(self):
    rec_dtype = np.dtype([('a', 'int32'), ('b', 'float64')], align=True)
    rec = from_dtype(rec_dtype)

    @cuda.jit((rec[:],))
    def foo(a):
        i = cuda.grid(1)
        a[i].a = a[i].b
    a_recarray = np.recarray(3, dtype=rec_dtype)
    for i in range(a_recarray.size):
        a_rec = a_recarray[i]
        a_rec.a = 0
        a_rec.b = (i + 1) * 123
    foo[1, 3](a_recarray)
    self.assertTrue(np.all(a_recarray.a == a_recarray.b))