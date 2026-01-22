from __future__ import division
from hypothesis import given, assume
from math import sqrt, floor
from blis.tests.common import *
from blis.py import gemm
@given(ndarrays(min_len=10, max_len=100, min_val=-100.0, max_val=100.0, dtype='float32'), ndarrays(min_len=10, max_len=100, min_val=-100.0, max_val=100.0, dtype='float32'), integers(min_value=2, max_value=1000), integers(min_value=2, max_value=1000), integers(min_value=2, max_value=1000))
def test_memoryview_float_notrans(A, B, a_rows, a_cols, out_cols):
    A, B, C = _reshape_for_gemm(A, B, a_rows, a_cols, out_cols, dtype='float32')
    assume(A is not None)
    assume(B is not None)
    assume(C is not None)
    assume(A.size >= 1)
    assume(B.size >= 1)
    assume(C.size >= 1)
    gemm(A, B, out=C)
    numpy_result = A.dot(B)
    assert_allclose(numpy_result, C, atol=0.001, rtol=0.001)