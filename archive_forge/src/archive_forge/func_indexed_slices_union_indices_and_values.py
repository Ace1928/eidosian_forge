import functools
import tensorflow as tf
def indexed_slices_union_indices_and_values(x1, x2_indices, x2_values=None):
    """Compute the indices for the union of two `tf.IndexedSlices` and modify
    the values for these indices.

    Args:
        x1: the first `tf.IndexedSlices`.
        x2_indices: the indices for the second `tf.IndexedSlices`.
        x2_value: (optional) the values for the second `tf.IndexedSlices`.
    Returns: A tuple containing:
        - the indices for the union
        - `x1` values for the union indices (some zeros were added)
        - `x2` values for the union indices (some zeros were added) or `None` if
          `x2_values` was `None`.
    """
    dim_0 = x1.dense_shape[0]
    x1_indices_expanded = tf.expand_dims(x1.indices, axis=1)
    x2_indices_expanded = tf.expand_dims(x2_indices, axis=1)
    x1_indices_count = tf.shape(x1_indices_expanded)[0]
    x2_indices_count = tf.shape(x2_indices_expanded)[0]
    x1_indices_one_hot = tf.scatter_nd(x1_indices_expanded, ones_bool((x1_indices_count,)), (dim_0,))
    x2_indices_one_hot = tf.scatter_nd(x2_indices_expanded, ones_bool((x2_indices_count,)), (dim_0,))
    union_indices = tf.squeeze(tf.where(tf.math.logical_or(x1_indices_one_hot, x2_indices_one_hot)), axis=-1)
    union_indices_count = tf.shape(union_indices)[0]

    def values_for_union(indices_expanded, indices_count, values):
        indices_indices = tf.scatter_nd(indices_expanded, tf.range(1, indices_count + 1), (dim_0,))
        to_union_indices = tf.gather(indices_indices, union_indices)
        values_with_leading_zeros = tf.concat([tf.zeros((1,) + values.shape[1:], values.dtype), values], axis=0)
        return tf.gather(values_with_leading_zeros, to_union_indices)
    x1_values_for_union_indices = tf.cond(tf.equal(x1_indices_count, union_indices_count), lambda: x1.values, lambda: values_for_union(x1_indices_expanded, x1_indices_count, x1.values))
    if x2_values is not None:
        x2_values_for_union_indices = tf.cond(tf.equal(x2_indices_count, union_indices_count), lambda: x2_values, lambda: values_for_union(x2_indices_expanded, x2_indices_count, x2_values))
    else:
        x2_values_for_union_indices = None
    return (union_indices, x1_values_for_union_indices, x2_values_for_union_indices)