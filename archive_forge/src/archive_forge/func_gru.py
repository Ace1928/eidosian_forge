import tensorflow as tf
import tree
from keras.src.utils.nest import pack_sequence_as
def gru(inputs, initial_state, mask, kernel, recurrent_kernel, bias, activation, recurrent_activation, return_sequences=False, go_backwards=False, unroll=False, time_major=False, reset_after=True):
    cudnn_supported = cudnn_ok(activation, recurrent_activation, unroll, use_bias=bias is not None, reset_after=reset_after)
    if not cudnn_supported or mask is not None:
        raise NotImplementedError
    from keras.src.backend.tensorflow import Variable
    if isinstance(kernel, Variable):
        kernel = kernel.value
    if isinstance(recurrent_kernel, Variable):
        recurrent_kernel = recurrent_kernel.value
    if isinstance(bias, Variable):
        bias = bias.value
    try:
        return _cudnn_gru(inputs, initial_state, kernel, recurrent_kernel, bias, mask, time_major, go_backwards, return_sequences)
    except tf.errors.InvalidArgumentError:
        raise NotImplementedError
    except tf.errors.NotFoundError:
        raise NotImplementedError