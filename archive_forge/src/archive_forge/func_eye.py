import builtins
import enum
import functools
import math
import numbers
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import manip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.ops.numpy_ops import np_utils
from tensorflow.python.util import nest
from tensorflow.python.util import tf_export
@tf_export.tf_export('experimental.numpy.eye', v1=[])
@np_utils.np_doc('eye')
def eye(N, M=None, k=0, dtype=float):
    if dtype:
        dtype = np_utils.result_type(dtype)
    if not M:
        M = N
    N = int(N)
    M = int(M)
    k = int(k)
    if k >= M or -k >= N:
        return zeros([N, M], dtype=dtype)
    if k == 0:
        return linalg_ops.eye(N, M, dtype=dtype)
    diag_len = builtins.min(N, M)
    if k > 0:
        if N >= M:
            diag_len -= k
        elif N + k > M:
            diag_len = M - k
    elif k <= 0:
        if M >= N:
            diag_len += k
        elif M - k > N:
            diag_len = N + k
    diagonal_ = array_ops.ones([diag_len], dtype=dtype)
    return array_ops.matrix_diag(diagonal=diagonal_, num_rows=N, num_cols=M, k=k)