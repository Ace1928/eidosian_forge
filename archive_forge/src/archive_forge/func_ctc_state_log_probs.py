import uuid
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gen_ctc_ops
from tensorflow.python.ops import inplace_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.nn_grad import _BroadcastMul
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def ctc_state_log_probs(seq_lengths, max_seq_length):
    """Computes CTC alignment initial and final state log probabilities.

  Create the initial/final state values directly as log values to avoid
  having to take a float64 log on tpu (which does not exist).

  Args:
    seq_lengths: int tensor of shape [batch_size], seq lengths in the batch.
    max_seq_length: int, max sequence length possible.

  Returns:
    initial_state_log_probs, final_state_log_probs
  """
    batch_size = _get_dim(seq_lengths, 0)
    num_label_states = max_seq_length + 1
    num_duration_states = 2
    num_states = num_duration_states * num_label_states
    log_0 = math_ops.cast(math_ops.log(math_ops.cast(0, dtypes.float64) + 1e-307), dtypes.float32)
    initial_state_log_probs = array_ops.one_hot(indices=array_ops.zeros([batch_size], dtype=dtypes.int32), depth=num_states, on_value=0.0, off_value=log_0, axis=1)
    label_final_state_mask = array_ops.one_hot(seq_lengths, depth=num_label_states, axis=0)
    duration_final_state_mask = array_ops.ones([num_duration_states, 1, batch_size])
    final_state_mask = duration_final_state_mask * label_final_state_mask
    final_state_log_probs = (1.0 - final_state_mask) * log_0
    final_state_log_probs = array_ops.reshape(final_state_log_probs, [num_states, batch_size])
    return (initial_state_log_probs, array_ops.transpose(final_state_log_probs))