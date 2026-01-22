import tensorflow as tf
import tree
from keras.src.utils.nest import pack_sequence_as
def cudnn_ok(activation, recurrent_activation, unroll, use_bias, reset_after=None):
    if reset_after is None:
        args_supported = _do_lstm_arguments_support_cudnn(activation=activation, recurrent_activation=recurrent_activation, unroll=unroll, use_bias=use_bias)
    else:
        args_supported = _do_gru_arguments_support_cudnn(activation=activation, recurrent_activation=recurrent_activation, unroll=unroll, use_bias=use_bias, reset_after=reset_after)
    return args_supported and _is_gpu_available()