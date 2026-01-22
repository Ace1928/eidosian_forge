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
def standard_lstm(inputs, init_h, init_c, kernel, recurrent_kernel, bias, mask, time_major, go_backwards, sequence_lengths, zero_output_for_mask, return_sequences):
    """LSTM with standard kernel implementation.

    This implementation can be run on all types for hardware.

    This implementation lifts out all the layer weights and make them function
    parameters. It has same number of tensor input params as the cuDNN
    counterpart. The RNN step logic has been simplified, eg dropout and mask is
    removed since cuDNN implementation does not support that.

    Note that the first half of the bias tensor should be ignored by this impl.
    The cuDNN impl need an extra set of input gate bias. In order to make the
    both function take same shape of parameter, that extra set of bias is also
    feed
    here.

    Args:
      inputs: input tensor of LSTM layer.
      init_h: initial state tensor for the cell output.
      init_c: initial state tensor for the cell hidden state.
      kernel: weights for cell kernel.
      recurrent_kernel: weights for cell recurrent kernel.
      bias: weights for cell kernel bias and recurrent bias. Only recurrent bias
        is used in this case.
      mask: Boolean tensor for mask out the steps within sequence.
        An individual `True` entry indicates that the corresponding timestep
        should be utilized, while a `False` entry indicates that the
        corresponding timestep should be ignored.
      time_major: boolean, whether the inputs are in the format of
        [time, batch, feature] or [batch, time, feature].
      go_backwards: Boolean (default False). If True, process the input sequence
        backwards and return the reversed sequence.
      sequence_lengths: The lengths of all sequences coming from a variable
        length input, such as ragged tensors. If the input has a fixed timestep
        size, this should be None.
      zero_output_for_mask: Boolean, whether to output zero for masked timestep.
      return_sequences: Boolean. If True, return the recurrent outputs for all
        timesteps in the sequence. If False, only return the output for the
        last timestep (which consumes less memory).

    Returns:
      last_output: output tensor for the last timestep, which has shape
        [batch, units].
      outputs:
        - If `return_sequences=True`: output tensor for all timesteps,
          which has shape [batch, time, units].
        - Else, a tensor equal to `last_output` with shape [batch, 1, units]
      state_0: the cell output, which has same shape as init_h.
      state_1: the cell hidden state, which has same shape as init_c.
      runtime: constant string tensor which indicate real runtime hardware. This
        value is for testing purpose and should be used by user.
    """
    input_shape = backend.int_shape(inputs)
    timesteps = input_shape[0] if time_major else input_shape[1]

    def step(cell_inputs, cell_states):
        """Step function that will be used by Keras RNN backend."""
        h_tm1 = cell_states[0]
        c_tm1 = cell_states[1]
        z = backend.dot(cell_inputs, kernel)
        z += backend.dot(h_tm1, recurrent_kernel)
        z = backend.bias_add(z, bias)
        z0, z1, z2, z3 = tf.split(z, 4, axis=1)
        i = tf.sigmoid(z0)
        f = tf.sigmoid(z1)
        c = f * c_tm1 + i * tf.tanh(z2)
        o = tf.sigmoid(z3)
        h = o * tf.tanh(c)
        return (h, [h, c])
    last_output, outputs, new_states = backend.rnn(step, inputs, [init_h, init_c], constants=None, unroll=False, time_major=time_major, mask=mask, go_backwards=go_backwards, input_length=sequence_lengths if sequence_lengths is not None else timesteps, zero_output_for_mask=zero_output_for_mask, return_all_outputs=return_sequences)
    return (last_output, outputs, new_states[0], new_states[1], gru_lstm_utils.runtime(gru_lstm_utils.RUNTIME_CPU))