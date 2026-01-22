import functools
import tensorflow as tf
def values_for_union(indices_expanded, indices_count, values):
    indices_indices = tf.scatter_nd(indices_expanded, tf.range(1, indices_count + 1), (dim_0,))
    to_union_indices = tf.gather(indices_indices, union_indices)
    values_with_leading_zeros = tf.concat([tf.zeros((1,) + values.shape[1:], values.dtype), values], axis=0)
    return tf.gather(values_with_leading_zeros, to_union_indices)