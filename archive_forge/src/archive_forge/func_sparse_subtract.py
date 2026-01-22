import functools
import tensorflow as tf
def sparse_subtract(x1, x2):
    """Subtraction for `tf.SparseTensor`s.

    Either `x1` or `x2` or both can be `tf.SparseTensor`s.

    Args:
        x1: fist tensor to add.
        x2: second tensor to add.
    Returns:
        The sum of `x1` and `x2`, which is a `tf.SparseTensor` if and only if
        both `x1` or `x2` are `tf.SparseTensor`s.
    """
    if isinstance(x2, tf.SparseTensor):
        return tf.sparse.add(x1, tf.sparse.map_values(tf.negative, x2))
    else:
        return tf.sparse.add(x1, tf.negative(x2))