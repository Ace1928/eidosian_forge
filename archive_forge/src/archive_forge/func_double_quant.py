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
def double_quant(A, col_stats=None, row_stats=None, out_col=None, out_row=None, threshold=0.0):
    device = A.device
    assert A.dtype == torch.half
    assert device.type == 'cuda'
    prev_device = pre_call(A.device)
    cols = A.shape[-1]
    if len(A.shape) == 3:
        rows = A.shape[0] * A.shape[1]
    else:
        rows = A.shape[0]
    if row_stats is None or col_stats is None:
        row_stats, col_stats, nnz_row_ptr = get_colrow_absmax(A, threshold=threshold)
    if out_col is None:
        out_col = torch.zeros(A.shape, device=device, dtype=torch.int8)
    if out_row is None:
        out_row = torch.zeros(A.shape, device=device, dtype=torch.int8)
    coo_tensor = None
    ptrA = get_ptr(A)
    ptrColStats = get_ptr(col_stats)
    ptrRowStats = get_ptr(row_stats)
    ptrOutCol = get_ptr(out_col)
    ptrOutRow = get_ptr(out_row)
    is_on_gpu([A, col_stats, row_stats, out_col, out_row])
    if threshold > 0.0:
        nnz = nnz_row_ptr[-1].item()
        if nnz > 0:
            coo_tensor = coo_zeros(A.shape[0], A.shape[1], nnz_row_ptr[-1].item(), device)
            ptrRowIdx = get_ptr(coo_tensor.rowidx)
            ptrColIdx = get_ptr(coo_tensor.colidx)
            ptrVal = get_ptr(coo_tensor.values)
            ptrRowPtr = get_ptr(nnz_row_ptr)
            lib.cdouble_rowcol_quant(ptrA, ptrRowStats, ptrColStats, ptrOutCol, ptrOutRow, ptrRowIdx, ptrColIdx, ptrVal, ptrRowPtr, ct.c_float(threshold), ct.c_int32(rows), ct.c_int32(cols))
            val, idx = torch.sort(coo_tensor.rowidx)
            coo_tensor.rowidx = val
            coo_tensor.colidx = coo_tensor.colidx[idx]
            coo_tensor.values = coo_tensor.values[idx]
        else:
            lib.cdouble_rowcol_quant(ptrA, ptrRowStats, ptrColStats, ptrOutCol, ptrOutRow, None, None, None, None, ct.c_float(0.0), ct.c_int32(rows), ct.c_int32(cols))
    else:
        lib.cdouble_rowcol_quant(ptrA, ptrRowStats, ptrColStats, ptrOutCol, ptrOutRow, None, None, None, None, ct.c_float(threshold), ct.c_int32(rows), ct.c_int32(cols))
    post_call(prev_device)
    return (out_row, out_col, row_stats, col_stats, coo_tensor)