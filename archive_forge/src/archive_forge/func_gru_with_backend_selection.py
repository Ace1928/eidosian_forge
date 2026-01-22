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
def gru_with_backend_selection(inputs, init_h, kernel, recurrent_kernel, bias, mask, time_major, go_backwards, sequence_lengths, zero_output_for_mask, return_sequences):
    """Call the GRU with optimized backend kernel selection.

    Under the hood, this function will create two TF function, one with the most
    generic kernel and can run on all device condition, and the second one with
    cuDNN specific kernel, which can only run on GPU.

    The first function will be called with normal_lstm_params, while the second
    function is not called, but only registered in the graph. The Grappler will
    do the proper graph rewrite and swap the optimized TF function based on the
    device placement.

    Args:
      inputs: Input tensor of GRU layer.
      init_h: Initial state tensor for the cell output.
      kernel: Weights for cell kernel.
      recurrent_kernel: Weights for cell recurrent kernel.
      bias: Weights for cell kernel bias and recurrent bias. Only recurrent bias
        is used in this case.
      mask: Boolean tensor for mask out the steps within sequence.
        An individual `True` entry indicates that the corresponding timestep
        should be utilized, while a `False` entry indicates that the
        corresponding timestep should be ignored.
      time_major: Boolean, whether the inputs are in the format of
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
      List of output tensors, same as standard_gru.
    """
    params = {'inputs': inputs, 'init_h': init_h, 'kernel': kernel, 'recurrent_kernel': recurrent_kernel, 'bias': bias, 'mask': mask, 'time_major': time_major, 'go_backwards': go_backwards, 'sequence_lengths': sequence_lengths, 'zero_output_for_mask': zero_output_for_mask, 'return_sequences': return_sequences}

    def gpu_gru_with_fallback(inputs, init_h, kernel, recurrent_kernel, bias, mask, time_major, go_backwards, sequence_lengths, zero_output_for_mask, return_sequences):
        """Use cuDNN kernel when mask is none or strictly right padded."""

        def cudnn_gru_fn():
            return gpu_gru(inputs=inputs, init_h=init_h, kernel=kernel, recurrent_kernel=recurrent_kernel, bias=bias, mask=mask, time_major=time_major, go_backwards=go_backwards, sequence_lengths=sequence_lengths, return_sequences=return_sequences)

        def standard_gru_fn():
            return standard_gru(inputs=inputs, init_h=init_h, kernel=kernel, recurrent_kernel=recurrent_kernel, bias=bias, mask=mask, time_major=time_major, go_backwards=go_backwards, sequence_lengths=sequence_lengths, zero_output_for_mask=zero_output_for_mask, return_sequences=return_sequences)
        return tf.__internal__.smart_cond.smart_cond(gru_lstm_utils.is_cudnn_supported_inputs(mask, time_major, sequence_lengths), true_fn=cudnn_gru_fn, false_fn=standard_gru_fn)
    if gru_lstm_utils.use_new_gru_lstm_impl():
        last_output, outputs, new_h, runtime = tf.__internal__.execute_fn_for_device({gru_lstm_utils.CPU_DEVICE_NAME: lambda: standard_gru(**params), gru_lstm_utils.GPU_DEVICE_NAME: lambda: gpu_gru_with_fallback(**params)}, lambda: standard_gru(**params))
    else:
        api_name = 'gru_' + str(uuid.uuid4())
        supportive_attribute = {'time_major': time_major, 'go_backwards': go_backwards}
        defun_standard_gru = gru_lstm_utils.generate_defun_backend(api_name, gru_lstm_utils.CPU_DEVICE_NAME, standard_gru, supportive_attribute)
        defun_gpu_gru = gru_lstm_utils.generate_defun_backend(api_name, gru_lstm_utils.GPU_DEVICE_NAME, gpu_gru_with_fallback, supportive_attribute)
        last_output, outputs, new_h, runtime = defun_standard_gru(**params)
        gru_lstm_utils.function_register(defun_gpu_gru, **params)
    return (last_output, outputs, new_h, runtime)