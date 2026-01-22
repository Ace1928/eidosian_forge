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
def cupy_csr2dense():
    return cupy.ElementwiseKernel('int32 M, int32 N, raw I INDPTR, I INDICES, T DATA, bool C_ORDER', 'raw T OUT', '\n        int row = get_row_id(i, 0, M - 1, &(INDPTR[0]));\n        int col = INDICES;\n        if (C_ORDER) {\n            OUT[col + N * row] += DATA;\n        } else {\n            OUT[row + M * col] += DATA;\n        }\n        ', 'cupyx_scipy_sparse_csr2dense', preamble=_GET_ROW_ID_)