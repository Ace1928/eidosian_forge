import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def inverse_iteration_step(i, v, nrm_v, nrm_v_old):
    v = tridiagonal_solve(diags, v, diagonals_format='sequence', partial_pivoting=True, perturb_singular=True)
    nrm_v_old = nrm_v
    nrm_v = norm(v, axis=1)
    v = v / nrm_v[:, array_ops.newaxis]
    v = orthogonalize_close_eigenvectors(v)
    return (i + 1, v, nrm_v, nrm_v_old)