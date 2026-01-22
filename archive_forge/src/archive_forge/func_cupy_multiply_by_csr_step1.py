import operator
import warnings
import numpy
import cupy
from cupy._core import _accelerator
from cupy.cuda import cub
from cupy.cuda import runtime
from cupyx import cusparse
from cupyx.scipy.sparse import _base
from cupyx.scipy.sparse import _compressed
from cupyx.scipy.sparse import _csc
from cupyx.scipy.sparse import SparseEfficiencyWarning
from cupyx.scipy.sparse import _util
@cupy._util.memoize(for_each_device=True)
def cupy_multiply_by_csr_step1():
    return cupy.ElementwiseKernel('\n        raw A A_DATA, raw I A_INDPTR, raw I A_INDICES, int32 A_M, int32 A_N,\n        raw B B_DATA, raw I B_INDPTR, raw I B_INDICES, int32 B_M, int32 B_N,\n        raw I C_INDPTR, int32 C_M, int32 C_N\n        ', 'C C_DATA, I C_INDICES, raw I FLAGS, raw I NNZ_EACH_ROW', '\n        int i_c = i;\n        int m_c = get_row_id(i_c, 0, C_M - 1, &(C_INDPTR[0]));\n\n        int i_a = i;\n        if (C_M > A_M && A_M == 1) {\n            i_a -= C_INDPTR[m_c];\n        }\n        if (C_N > A_N && A_N == 1) {\n            i_a /= C_N;\n        }\n        int n_c = A_INDICES[i_a];\n        if (C_N > A_N && A_N == 1) {\n            n_c = i % C_N;\n        }\n        int m_b = m_c;\n        if (C_M > B_M && B_M == 1) {\n            m_b = 0;\n        }\n        int n_b = n_c;\n        if (C_N > B_N && B_N == 1) {\n            n_b = 0;\n        }\n        int i_b = find_index_holding_col_in_row(m_b, n_b,\n            &(B_INDPTR[0]), &(B_INDICES[0]));\n        if (i_b >= 0) {\n            atomicAdd(&(NNZ_EACH_ROW[m_c+1]), 1);\n            FLAGS[i+1] = 1;\n            C_DATA = (C)(A_DATA[i_a] * B_DATA[i_b]);\n            C_INDICES = n_c;\n        }\n        ', 'cupyx_scipy_sparse_csr_multiply_by_csr_step1', preamble=_GET_ROW_ID_ + _FIND_INDEX_HOLDING_COL_IN_ROW_)