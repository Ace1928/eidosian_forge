import functools
import tensorflow as tf
def sparse_union_indices_and_values(x1, x2_indices, x2_values=None):
    """Compute the indices for the union of the indices of the provided
    `tf.SparseTensor`s and another set of indices and return the modified values
    for these indices.

    Args:
        x: a `tf.SparseTensor`.
        indices: another set of indices in the `tf.SparseTensor` format.
    Returns: A tuple containing:
        - the indices for the union
        - `x1` values for the union indices (some zeros were added)
        - `x2` values for the union indices (some zeros were added) or `None` if
          `x2_values` was `None`.
    """
    zeros2 = tf.SparseTensor(x2_indices, tf.zeros((tf.shape(x2_indices)[0],), x1.values.dtype), x1.dense_shape)
    x1_for_union = tf.sparse.add(x1, zeros2)
    if x2_values is not None:
        x2 = tf.SparseTensor(x2_indices, x2_values, x1.dense_shape)
        zeros1 = tf.sparse.map_values(tf.zeros_like, x1)
        x2_for_union = tf.sparse.add(x2, zeros1)
        return (x1_for_union.indices, x1_for_union.values, x2_for_union.values)
    else:
        return (x1_for_union.indices, x1_for_union.values, None)