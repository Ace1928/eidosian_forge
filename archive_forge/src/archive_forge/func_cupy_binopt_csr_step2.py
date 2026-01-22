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
def cupy_binopt_csr_step2(op_name):
    name = 'cupyx_scipy_sparse_csr_binopt' + op_name + 'step2'
    return cupy.ElementwiseKernel('\n        raw I A_INFO, raw B A_VALID, raw I A_TMP_INDICES, raw O A_TMP_DATA,\n        int32 A_NNZ,\n        raw I B_INFO, raw B B_VALID, raw I B_TMP_INDICES, raw O B_TMP_DATA,\n        int32 B_NNZ\n        ', 'raw I C_INDICES, raw O C_DATA', '\n        if (i < A_NNZ) {\n            int j = i;\n            if (A_VALID[j]) {\n                C_INDICES[A_INFO[j]] = A_TMP_INDICES[j];\n                C_DATA[A_INFO[j]]    = A_TMP_DATA[j];\n            }\n        } else if (i < A_NNZ + B_NNZ) {\n            int j = i - A_NNZ;\n            if (B_VALID[j]) {\n                C_INDICES[B_INFO[j]] = B_TMP_INDICES[j];\n                C_DATA[B_INFO[j]]    = B_TMP_DATA[j];\n            }\n        }\n        ', name)