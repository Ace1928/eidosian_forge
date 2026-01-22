import ctypes as ct
from functools import reduce  # Required in Python 3
import itertools
import operator
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
from torch import Tensor
from bitsandbytes.utils import pack_dict_to_tensor, unpack_tensor_to_dict
from .cextension import COMPILED_WITH_CUDA, lib
def igemmlt(A, B, SA, SB, out=None, Sout=None, dtype=torch.int32):
    shapeA = SA[0]
    shapeB = SB[0]
    dimsA = len(shapeA)
    dimsB = len(shapeB)
    assert dimsB == 2, 'Only two dimensional matrices are supported for argument B'
    if dimsA == 2:
        m = shapeA[0]
    elif dimsA == 3:
        m = shapeA[0] * shapeA[1]
    rows = n = shapeB[0]
    assert prod(list(shapeA)) > 0, f'Input tensor dimensions need to be > 0: {shapeA}'
    if shapeA[0] == 0 and dimsA == 2:
        return torch.empty((0, shapeB[0]), device=A.device, dtype=torch.float16)
    elif shapeA[1] == 0 and dimsA == 3:
        return torch.empty(tuple(shapeA[:2] + [shapeB[0]]), device=A.device, dtype=torch.float16)
    if dimsA == 2 and out is None:
        out, Sout = get_transform_buffer((shapeA[0], shapeB[0]), dtype, A.device, 'col32', 'row')
    elif dimsA == 3 and out is None:
        out, Sout = get_transform_buffer((shapeA[0], shapeA[1], shapeB[0]), dtype, A.device, 'col32', 'row')
    assert dimsB != 3, 'len(B.shape)==3 not supported'
    assert A.device.type == 'cuda'
    assert B.device.type == 'cuda'
    assert A.dtype == torch.int8
    assert B.dtype == torch.int8
    assert out.dtype == dtype
    assert SA[1] == 'col32'
    assert SB[1] in ['col_turing', 'col_ampere']
    assert Sout[1] == 'col32'
    assert shapeA[-1] == shapeB[-1], f'Matmullt only supports A @ B^T. Inner matrix dimensions do not match: A @ B = {shapeA} @ {shapeB}'
    formatB = SB[1]
    prev_device = A.device
    torch.cuda.set_device(A.device)
    ptr = CUBLAS_Context.get_instance().get_context(A.device)
    ptrA = get_ptr(A)
    ptrB = get_ptr(B)
    ptrC = get_ptr(out)
    k = shapeA[-1]
    lda = ct.c_int32(m * 32)
    if formatB == 'col_turing':
        ldb = ct.c_int32((rows + 7) // 8 * 8 * 32)
    else:
        ldb = ct.c_int32((rows + 31) // 32 * 32 * 32)
    ldc = ct.c_int32(m * 32)
    m = ct.c_int32(m)
    n = ct.c_int32(n)
    k = ct.c_int32(k)
    has_error = 0
    ptrRowScale = get_ptr(None)
    is_on_gpu([A, B, out])
    if formatB == 'col_turing':
        if dtype == torch.int32:
            has_error = lib.cigemmlt_turing_32(ptr, m, n, k, ptrA, ptrB, ptrC, ptrRowScale, lda, ldb, ldc)
        else:
            has_error = lib.cigemmlt_turing_8(ptr, m, n, k, ptrA, ptrB, ptrC, ptrRowScale, lda, ldb, ldc)
    elif formatB == 'col_ampere':
        if dtype == torch.int32:
            has_error = lib.cigemmlt_ampere_32(ptr, m, n, k, ptrA, ptrB, ptrC, ptrRowScale, lda, ldb, ldc)
        else:
            has_error = lib.cigemmlt_ampere_8(ptr, m, n, k, ptrA, ptrB, ptrC, ptrRowScale, lda, ldb, ldc)
    if has_error == 100:
        raise NotImplementedError('igemmlt not available (probably built with NO_CUBLASLT)')
    if has_error:
        print(f'A: {shapeA}, B: {shapeB}, C: {Sout[0]}; (lda, ldb, ldc): {(lda, ldb, ldc)}; (m, n, k): {(m, n, k)}')
        raise Exception('cublasLt ran into an error!')
    torch.cuda.set_device(prev_device)
    return (out, Sout)