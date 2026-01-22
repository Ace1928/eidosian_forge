from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import gen_math_ops
def not_eq(a, b):
    """Functional form of "not-equal"."""
    return not_(eq(a, b))