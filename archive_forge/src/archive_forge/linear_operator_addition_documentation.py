import abc
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_diag
from tensorflow.python.ops.linalg import linear_operator_full_matrix
from tensorflow.python.ops.linalg import linear_operator_identity
from tensorflow.python.ops.linalg import linear_operator_lower_triangular
Return new `LinearOperator` acting like `op1 + op2`.

    Args:
      op1:  `LinearOperator`
      op2:  `LinearOperator`, with `shape` and `dtype` such that adding to
        `op1` is allowed.
      operator_name:  `String` name to give to returned `LinearOperator`
      hints:  `_Hints` object.  Returned `LinearOperator` will be created with
        these hints.

    Returns:
      `LinearOperator`
    