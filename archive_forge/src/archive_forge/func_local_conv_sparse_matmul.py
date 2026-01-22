import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.utils import conv_utils
def local_conv_sparse_matmul(inputs, kernel, kernel_idxs, kernel_shape, output_shape):
    """Apply N-D convolution with unshared weights using a single sparse matmul.

    This method outputs `inputs . tf.sparse.SparseTensor(indices=kernel_idxs,
    values=kernel, dense_shape=kernel_shape)`, with `.` standing for
    matrix-multiply. It also reshapes `inputs` to 2-D and `output` to (N+2)-D.

    Args:
        inputs: (N+2)-D tensor with shape `(batch_size, channels_in, d_in1, ...,
          d_inN)` or `(batch_size, d_in1, ..., d_inN, channels_in)`.
        kernel: a 1-D tensor with shape `(len(kernel_idxs),)` containing all the
          weights of the layer.
        kernel_idxs:  a list of integer tuples representing indices in a sparse
          matrix performing the un-shared convolution as a matrix-multiply.
        kernel_shape: a tuple `(input_size, output_size)`, where `input_size =
          channels_in * d_in1 * ... * d_inN` and `output_size = channels_out *
          d_out1 * ... * d_outN`.
        output_shape: a tuple of (N+2) elements representing the output shape:
          `(batch_size, channels_out, d_out1, ..., d_outN)` or `(batch_size,
          d_out1, ..., d_outN, channels_out)`, with the ordering of channels and
          spatial dimensions matching that of the input.

    Returns:
        Output (N+2)-D dense tensor with shape `output_shape`.
    """
    inputs_flat = backend.reshape(inputs, (backend.shape(inputs)[0], -1))
    output_flat = tf.sparse.sparse_dense_matmul(sp_a=tf.SparseTensor(kernel_idxs, kernel, kernel_shape), b=inputs_flat, adjoint_b=True)
    output_flat_transpose = backend.transpose(output_flat)
    output_reshaped = backend.reshape(output_flat_transpose, [backend.shape(output_flat_transpose)[0]] + output_shape.as_list()[1:])
    return output_reshaped