import uuid
import tensorflow.compat.v2 as tf
from keras.src import activations
from keras.src import backend
from keras.src import constraints
from keras.src import initializers
from keras.src import regularizers
from keras.src.engine import base_layer
from keras.src.engine.input_spec import InputSpec
from keras.src.layers.rnn import gru_lstm_utils
from keras.src.layers.rnn import rnn_utils
from keras.src.layers.rnn.base_rnn import RNN
from keras.src.layers.rnn.dropout_rnn_cell_mixin import DropoutRNNCellMixin
from keras.src.utils import tf_utils
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
def gpu_lstm(inputs, init_h, init_c, kernel, recurrent_kernel, bias, mask, time_major, go_backwards, sequence_lengths, return_sequences):
    """LSTM with either cuDNN or ROCm implementation which is only available for
    GPU.

    Note that currently only right padded data is supported, or the result will
    be polluted by the unmasked data which should be filtered.

    Args:
      inputs: Input tensor of LSTM layer.
      init_h: Initial state tensor for the cell output.
      init_c: Initial state tensor for the cell hidden state.
      kernel: Weights for cell kernel.
      recurrent_kernel: Weights for cell recurrent kernel.
      bias: Weights for cell kernel bias and recurrent bias. Only recurrent bias
        is used in this case.
      mask: Boolean tensor for mask out the steps within sequence. An individual
        `True` entry indicates that the corresponding timestep should be
        utilized, while a `False` entry indicates that the corresponding
        timestep should be ignored.
      time_major: Boolean, whether the inputs are in the format of [time, batch,
        feature] or [batch, time, feature].
      go_backwards: Boolean (default False). If True, process the input sequence
        backwards and return the reversed sequence.
      sequence_lengths: The lengths of all sequences coming from a variable
        length input, such as ragged tensors. If the input has a fixed timestep
        size, this should be None.
      return_sequences: Boolean. If True, return the recurrent outputs for all
        timesteps in the sequence. If False, only return the output for the
        last timestep, matching the CPU function output format.

    Returns:
      last_output: Output tensor for the last timestep, which has shape
        [batch, units].
      outputs:
        - If `return_sequences=True`: output tensor for all timesteps,
          which has shape [batch, time, units].
        - Else, a tensor equal to `last_output` with shape [batch, 1, units]
      state_0: The cell output, which has same shape as init_h.
      state_1: The cell hidden state, which has same shape as init_c.
      runtime: Constant string tensor which indicate real runtime hardware. This
        value is for testing purpose and should not be used by user.
    """
    if mask is not None:
        sequence_lengths = gru_lstm_utils.calculate_sequence_by_mask(mask, time_major)
    if not time_major and sequence_lengths is None:
        inputs = tf.transpose(inputs, perm=(1, 0, 2))
        seq_axis, batch_axis = (0, 1)
    else:
        seq_axis, batch_axis = (0, 1) if time_major else (1, 0)
    init_h = tf.expand_dims(init_h, axis=seq_axis)
    init_c = tf.expand_dims(init_c, axis=seq_axis)
    weights = tf.split(kernel, 4, axis=1)
    weights += tf.split(recurrent_kernel, 4, axis=1)
    full_bias = tf.concat((tf.zeros_like(bias), bias), 0)
    if tf.sysconfig.get_build_info()['is_rocm_build']:
        weights = [weights[x] for x in (0, 1, 3, 2, 4, 5, 7, 6)]
        full_bias = tf.split(full_bias, 8, axis=0)
        full_bias = [full_bias[x] for x in (0, 1, 3, 2, 4, 5, 7, 6)]
    params = gru_lstm_utils.canonical_to_params(weights=weights, biases=tf.split(full_bias, 8), shape=tf.constant([-1]), transpose_weights=True)
    if sequence_lengths is not None:
        if go_backwards:
            inputs = tf.reverse_sequence(inputs, sequence_lengths, seq_axis=seq_axis, batch_axis=batch_axis)
        outputs, h, c, _, _ = tf.raw_ops.CudnnRNNV3(input=inputs, input_h=init_h, input_c=init_c, params=params, is_training=True, rnn_mode='lstm', sequence_lengths=sequence_lengths, time_major=time_major)
        if go_backwards:
            outputs = tf.reverse_sequence(outputs, sequence_lengths, seq_axis=seq_axis, batch_axis=batch_axis)
            outputs = tf.reverse(outputs, axis=[seq_axis])
    else:
        if go_backwards:
            inputs = tf.reverse(inputs, axis=[0])
        outputs, h, c, _ = tf.raw_ops.CudnnRNN(input=inputs, input_h=init_h, input_c=init_c, params=params, is_training=True, rnn_mode='lstm')
    last_output = outputs[-1]
    if not time_major and sequence_lengths is None and return_sequences:
        outputs = tf.transpose(outputs, perm=[1, 0, 2])
    h = tf.squeeze(h, axis=seq_axis)
    c = tf.squeeze(c, axis=seq_axis)
    if sequence_lengths is not None:
        last_output = h
    if not return_sequences:
        outputs = tf.expand_dims(last_output, axis=0 if time_major else 1)
    return (last_output, outputs, h, c, gru_lstm_utils.runtime(gru_lstm_utils.RUNTIME_GPU))