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
def cupy_multiply_by_dense():
    return cupy.ElementwiseKernel('\n        raw S SP_DATA, raw I SP_INDPTR, raw I SP_INDICES,\n        int32 SP_M, int32 SP_N,\n        raw D DN_DATA, int32 DN_M, int32 DN_N,\n        raw I OUT_INDPTR, int32 OUT_M, int32 OUT_N\n        ', 'O OUT_DATA, I OUT_INDICES', '\n        int i_out = i;\n        int m_out = get_row_id(i_out, 0, OUT_M - 1, &(OUT_INDPTR[0]));\n        int i_sp = i_out;\n        if (OUT_M > SP_M && SP_M == 1) {\n            i_sp -= OUT_INDPTR[m_out];\n        }\n        if (OUT_N > SP_N && SP_N == 1) {\n            i_sp /= OUT_N;\n        }\n        int n_out = SP_INDICES[i_sp];\n        if (OUT_N > SP_N && SP_N == 1) {\n            n_out = i_out - OUT_INDPTR[m_out];\n        }\n        int m_dn = m_out;\n        if (OUT_M > DN_M && DN_M == 1) {\n            m_dn = 0;\n        }\n        int n_dn = n_out;\n        if (OUT_N > DN_N && DN_N == 1) {\n            n_dn = 0;\n        }\n        OUT_DATA = (O)(SP_DATA[i_sp] * DN_DATA[n_dn + (DN_N * m_dn)]);\n        OUT_INDICES = n_out;\n        ', 'cupyx_scipy_sparse_csr_multiply_by_dense', preamble=_GET_ROW_ID_)