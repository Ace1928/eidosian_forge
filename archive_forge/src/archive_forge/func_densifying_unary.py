import functools
import tensorflow as tf
def densifying_unary(default_value):
    """Decorator to add support for `tf.SparseTensor` and `tf.IndexedSlices` to
    a non-zero-preserving element-wise unary operator.

    There are requirements on the operator for this decorator to work correctly:

    - The operator must be element-wise
    - The operator must be unary (one input tensor and one output tensor)
    - The operator must return a tensor of the same shape.

    Additional arguments to the function (besides the input tensor) are
    supported. The returned result is a dense tensor and contains
    `default_value` outside of the indices of the input tensor.

    Args:
        default_value: The value to use outside of indices. It must be the value
        that the operator returns for zero values.
    Returns:
        Wrapped function that supports `tf.SparseTensor` and `tf.IndexedSlices`.
    """

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
    return wrap_densifying_unary