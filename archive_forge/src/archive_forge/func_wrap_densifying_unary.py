import functools
import tensorflow as tf
def wrap_densifying_unary(func):

    @functools.wraps(func)
    def sparse_wrapper(x, *args, **kwargs):
        if isinstance(x, tf.SparseTensor):
            sparse_output = sparse_with_values(x, func(x.values, *args, **kwargs))
            return sparse_to_dense(sparse_output, tf.cast(default_value, sparse_output.values.dtype))
        elif isinstance(x, tf.IndexedSlices):
            sparse_output_values = func(x.values, *args, **kwargs)
            output = tf.fill(x.dense_shape, tf.cast(default_value, sparse_output_values.dtype))
            return tf.tensor_scatter_nd_update(output, tf.expand_dims(x.indices, 1), sparse_output_values)
        return func(x, *args, **kwargs)
    return sparse_wrapper