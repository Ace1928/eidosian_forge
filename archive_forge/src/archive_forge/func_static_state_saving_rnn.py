from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import control_flow_util_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import while_loop
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
@deprecation.deprecated(None, 'Please use `keras.layers.RNN(cell, stateful=True)`, which is equivalent to this API')
@tf_export(v1=['nn.static_state_saving_rnn'])
@dispatch.add_dispatch_support
def static_state_saving_rnn(cell, inputs, state_saver, state_name, sequence_length=None, scope=None):
    """RNN that accepts a state saver for time-truncated RNN calculation.

  Args:
    cell: An instance of `RNNCell`.
    inputs: A length T list of inputs, each a `Tensor` of shape `[batch_size,
      input_size]`.
    state_saver: A state saver object with methods `state` and `save_state`.
    state_name: Python string or tuple of strings.  The name to use with the
      state_saver. If the cell returns tuples of states (i.e., `cell.state_size`
      is a tuple) then `state_name` should be a tuple of strings having the same
      length as `cell.state_size`.  Otherwise it should be a single string.
    sequence_length: (optional) An int32/int64 vector size [batch_size]. See the
      documentation for rnn() for more details about sequence_length.
    scope: VariableScope for the created subgraph; defaults to "rnn".

  Returns:
    A pair (outputs, state) where:
      outputs is a length T list of outputs (one for each input)
      states is the final state

  Raises:
    TypeError: If `cell` is not an instance of RNNCell.
    ValueError: If `inputs` is `None` or an empty list, or if the arity and
     type of `state_name` does not match that of `cell.state_size`.
  """
    state_size = cell.state_size
    state_is_tuple = nest.is_nested(state_size)
    state_name_tuple = nest.is_nested(state_name)
    if state_is_tuple != state_name_tuple:
        raise ValueError(f'Argument `state_name` should be the same type as `cell.state_size`. Received: state_name={state_name!s}, cell.state_size={state_size!s}.')
    if state_is_tuple:
        state_name_flat = nest.flatten(state_name)
        state_size_flat = nest.flatten(state_size)
        if len(state_name_flat) != len(state_size_flat):
            raise ValueError(f'Number of elements in argument `state_name` and `cell.state_size` are mismatched. Received state_name={state_name} with {len(state_name_flat)} elements and cell.state_size={cell.state_size} with {len(state_size_flat)} elements.')
        initial_state = nest.pack_sequence_as(structure=state_size, flat_sequence=[state_saver.state(s) for s in state_name_flat])
    else:
        initial_state = state_saver.state(state_name)
    outputs, state = static_rnn(cell, inputs, initial_state=initial_state, sequence_length=sequence_length, scope=scope)
    if state_is_tuple:
        flat_state = nest.flatten(state)
        state_name = nest.flatten(state_name)
        save_state = [state_saver.save_state(name, substate) for name, substate in zip(state_name, flat_state)]
    else:
        save_state = [state_saver.save_state(state_name, state)]
    with ops.control_dependencies(save_state):
        last_output = outputs[-1]
        flat_last_output = nest.flatten(last_output)
        flat_last_output = [array_ops.identity(output) for output in flat_last_output]
        outputs[-1] = nest.pack_sequence_as(structure=last_output, flat_sequence=flat_last_output)
        if state_is_tuple:
            state = nest.pack_sequence_as(structure=state, flat_sequence=[array_ops.identity(s) for s in flat_state])
        else:
            state = array_ops.identity(state)
    return (outputs, state)