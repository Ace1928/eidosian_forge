import collections
import itertools
import json
import os
import sys
import threading
import warnings
import weakref
import numpy as np
from tensorflow.core.protobuf import config_pb2
from tensorflow.python import tf2
from tensorflow.python.client import session as session_module
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.eager.context import get_config
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device_spec
from tensorflow.python.framework import dtypes as dtypes_module
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.distribute import distribute_coordinator_utils as dc
from tensorflow.python.keras.engine import keras_tensor
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.keras.utils import object_identity
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gradients as gradients_module
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import map_fn as map_fn_lib
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import moving_averages
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.tools.docs import doc_controls
@dispatch.add_dispatch_support
def rnn(step_function, inputs, initial_states, go_backwards=False, mask=None, constants=None, unroll=False, input_length=None, time_major=False, zero_output_for_mask=False):
    """Iterates over the time dimension of a tensor.

  Args:
      step_function: RNN step function.
          Args;
              input; Tensor with shape `(samples, ...)` (no time dimension),
                  representing input for the batch of samples at a certain
                  time step.
              states; List of tensors.
          Returns;
              output; Tensor with shape `(samples, output_dim)`
                  (no time dimension).
              new_states; List of tensors, same length and shapes
                  as 'states'. The first state in the list must be the
                  output tensor at the previous timestep.
      inputs: Tensor of temporal data of shape `(samples, time, ...)`
          (at least 3D), or nested tensors, and each of which has shape
          `(samples, time, ...)`.
      initial_states: Tensor with shape `(samples, state_size)`
          (no time dimension), containing the initial values for the states used
          in the step function. In the case that state_size is in a nested
          shape, the shape of initial_states will also follow the nested
          structure.
      go_backwards: Boolean. If True, do the iteration over the time
          dimension in reverse order and return the reversed sequence.
      mask: Binary tensor with shape `(samples, time, 1)`,
          with a zero for every element that is masked.
      constants: List of constant values passed at each step.
      unroll: Whether to unroll the RNN or to use a symbolic `while_loop`.
      input_length: An integer or a 1-D Tensor, depending on whether
          the time dimension is fixed-length or not. In case of variable length
          input, it is used for masking in case there's no mask specified.
      time_major: Boolean. If true, the inputs and outputs will be in shape
          `(timesteps, batch, ...)`, whereas in the False case, it will be
          `(batch, timesteps, ...)`. Using `time_major = True` is a bit more
          efficient because it avoids transposes at the beginning and end of the
          RNN calculation. However, most TensorFlow data is batch-major, so by
          default this function accepts input and emits output in batch-major
          form.
      zero_output_for_mask: Boolean. If True, the output for masked timestep
          will be zeros, whereas in the False case, output from previous
          timestep is returned.

  Returns:
      A tuple, `(last_output, outputs, new_states)`.
          last_output: the latest output of the rnn, of shape `(samples, ...)`
          outputs: tensor with shape `(samples, time, ...)` where each
              entry `outputs[s, t]` is the output of the step function
              at time `t` for sample `s`.
          new_states: list of tensors, latest states returned by
              the step function, of shape `(samples, ...)`.

  Raises:
      ValueError: if input dimension is less than 3.
      ValueError: if `unroll` is `True` but input timestep is not a fixed
      number.
      ValueError: if `mask` is provided (not `None`) but states is not provided
          (`len(states)` == 0).
  """

    def swap_batch_timestep(input_t):
        axes = list(range(len(input_t.shape)))
        axes[0], axes[1] = (1, 0)
        return array_ops.transpose(input_t, axes)
    if not time_major:
        inputs = nest.map_structure(swap_batch_timestep, inputs)
    flatted_inputs = nest.flatten(inputs)
    time_steps = flatted_inputs[0].shape[0]
    batch = flatted_inputs[0].shape[1]
    time_steps_t = array_ops.shape(flatted_inputs[0])[0]
    for input_ in flatted_inputs:
        input_.shape.with_rank_at_least(3)
    if mask is not None:
        if mask.dtype != dtypes_module.bool:
            mask = math_ops.cast(mask, dtypes_module.bool)
        if len(mask.shape) == 2:
            mask = expand_dims(mask)
        if not time_major:
            mask = swap_batch_timestep(mask)
    if constants is None:
        constants = []

    def _expand_mask(mask_t, input_t, fixed_dim=1):
        if nest.is_nested(mask_t):
            raise ValueError('mask_t is expected to be tensor, but got %s' % mask_t)
        if nest.is_nested(input_t):
            raise ValueError('input_t is expected to be tensor, but got %s' % input_t)
        rank_diff = len(input_t.shape) - len(mask_t.shape)
        for _ in range(rank_diff):
            mask_t = array_ops.expand_dims(mask_t, -1)
        multiples = [1] * fixed_dim + input_t.shape.as_list()[fixed_dim:]
        return array_ops.tile(mask_t, multiples)
    if unroll:
        if not time_steps:
            raise ValueError('Unrolling requires a fixed number of timesteps.')
        states = tuple(initial_states)
        successive_states = []
        successive_outputs = []

        def _process_single_input_t(input_t):
            input_t = array_ops_stack.unstack(input_t)
            if go_backwards:
                input_t.reverse()
            return input_t
        if nest.is_nested(inputs):
            processed_input = nest.map_structure(_process_single_input_t, inputs)
        else:
            processed_input = (_process_single_input_t(inputs),)

        def _get_input_tensor(time):
            inp = [t_[time] for t_ in processed_input]
            return nest.pack_sequence_as(inputs, inp)
        if mask is not None:
            mask_list = array_ops_stack.unstack(mask)
            if go_backwards:
                mask_list.reverse()
            for i in range(time_steps):
                inp = _get_input_tensor(i)
                mask_t = mask_list[i]
                output, new_states = step_function(inp, tuple(states) + tuple(constants))
                tiled_mask_t = _expand_mask(mask_t, output)
                if not successive_outputs:
                    prev_output = zeros_like(output)
                else:
                    prev_output = successive_outputs[-1]
                output = array_ops.where_v2(tiled_mask_t, output, prev_output)
                flat_states = nest.flatten(states)
                flat_new_states = nest.flatten(new_states)
                tiled_mask_t = tuple((_expand_mask(mask_t, s) for s in flat_states))
                flat_final_states = tuple((array_ops.where_v2(m, s, ps) for m, s, ps in zip(tiled_mask_t, flat_new_states, flat_states)))
                states = nest.pack_sequence_as(states, flat_final_states)
                successive_outputs.append(output)
                successive_states.append(states)
            last_output = successive_outputs[-1]
            new_states = successive_states[-1]
            outputs = array_ops_stack.stack(successive_outputs)
            if zero_output_for_mask:
                last_output = array_ops.where_v2(_expand_mask(mask_list[-1], last_output), last_output, zeros_like(last_output))
                outputs = array_ops.where_v2(_expand_mask(mask, outputs, fixed_dim=2), outputs, zeros_like(outputs))
        else:
            for i in range(time_steps):
                inp = _get_input_tensor(i)
                output, states = step_function(inp, tuple(states) + tuple(constants))
                successive_outputs.append(output)
                successive_states.append(states)
            last_output = successive_outputs[-1]
            new_states = successive_states[-1]
            outputs = array_ops_stack.stack(successive_outputs)
    else:
        states = tuple(initial_states)
        input_ta = tuple((tensor_array_ops.TensorArray(dtype=inp.dtype, size=time_steps_t, tensor_array_name='input_ta_%s' % i) for i, inp in enumerate(flatted_inputs)))
        input_ta = tuple((ta.unstack(input_) if not go_backwards else ta.unstack(reverse(input_, 0)) for ta, input_ in zip(input_ta, flatted_inputs)))
        input_time_zero = nest.pack_sequence_as(inputs, [inp[0] for inp in flatted_inputs])
        output_time_zero, _ = step_function(input_time_zero, tuple(initial_states) + tuple(constants))
        output_ta = tuple((tensor_array_ops.TensorArray(dtype=out.dtype, size=time_steps_t, element_shape=out.shape, tensor_array_name='output_ta_%s' % i) for i, out in enumerate(nest.flatten(output_time_zero))))
        time = constant_op.constant(0, dtype='int32', name='time')
        if not context.executing_eagerly() and control_flow_util.GraphOrParentsInXlaContext(ops.get_default_graph()):
            max_iterations = math_ops.reduce_max(input_length)
        else:
            max_iterations = None
        while_loop_kwargs = {'cond': lambda time, *_: time < time_steps_t, 'maximum_iterations': max_iterations, 'parallel_iterations': 32, 'swap_memory': True}
        if mask is not None:
            if go_backwards:
                mask = reverse(mask, 0)
            mask_ta = tensor_array_ops.TensorArray(dtype=dtypes_module.bool, size=time_steps_t, tensor_array_name='mask_ta')
            mask_ta = mask_ta.unstack(mask)

            def masking_fn(time):
                return mask_ta.read(time)

            def compute_masked_output(mask_t, flat_out, flat_mask):
                tiled_mask_t = tuple((_expand_mask(mask_t, o, fixed_dim=len(mask_t.shape)) for o in flat_out))
                return tuple((array_ops.where_v2(m, o, fm) for m, o, fm in zip(tiled_mask_t, flat_out, flat_mask)))
        elif isinstance(input_length, tensor_lib.Tensor):
            if go_backwards:
                max_len = math_ops.reduce_max(input_length, axis=0)
                rev_input_length = math_ops.subtract(max_len - 1, input_length)

                def masking_fn(time):
                    return math_ops.less(rev_input_length, time)
            else:

                def masking_fn(time):
                    return math_ops.greater(input_length, time)

            def compute_masked_output(mask_t, flat_out, flat_mask):
                return tuple((array_ops.where(mask_t, o, zo) for o, zo in zip(flat_out, flat_mask)))
        else:
            masking_fn = None
        if masking_fn is not None:
            flat_zero_output = tuple((array_ops.zeros_like(o) for o in nest.flatten(output_time_zero)))

            def _step(time, output_ta_t, prev_output, *states):
                """RNN step function.

        Args:
            time: Current timestep value.
            output_ta_t: TensorArray.
            prev_output: tuple of outputs from time - 1.
            *states: List of states.

        Returns:
            Tuple: `(time + 1, output_ta_t, output) + tuple(new_states)`
        """
                current_input = tuple((ta.read(time) for ta in input_ta))
                current_input = nest.pack_sequence_as(inputs, current_input)
                mask_t = masking_fn(time)
                output, new_states = step_function(current_input, tuple(states) + tuple(constants))
                flat_output = nest.flatten(output)
                flat_mask_output = flat_zero_output if zero_output_for_mask else nest.flatten(prev_output)
                flat_new_output = compute_masked_output(mask_t, flat_output, flat_mask_output)
                flat_state = nest.flatten(states)
                flat_new_state = nest.flatten(new_states)
                for state, new_state in zip(flat_state, flat_new_state):
                    if isinstance(new_state, tensor_lib.Tensor):
                        new_state.set_shape(state.shape)
                flat_final_state = compute_masked_output(mask_t, flat_new_state, flat_state)
                new_states = nest.pack_sequence_as(new_states, flat_final_state)
                output_ta_t = tuple((ta.write(time, out) for ta, out in zip(output_ta_t, flat_new_output)))
                return (time + 1, output_ta_t, tuple(flat_new_output)) + tuple(new_states)
            final_outputs = while_loop.while_loop(body=_step, loop_vars=(time, output_ta, flat_zero_output) + states, **while_loop_kwargs)
            new_states = final_outputs[3:]
        else:

            def _step(time, output_ta_t, *states):
                """RNN step function.

        Args:
            time: Current timestep value.
            output_ta_t: TensorArray.
            *states: List of states.

        Returns:
            Tuple: `(time + 1,output_ta_t) + tuple(new_states)`
        """
                current_input = tuple((ta.read(time) for ta in input_ta))
                current_input = nest.pack_sequence_as(inputs, current_input)
                output, new_states = step_function(current_input, tuple(states) + tuple(constants))
                flat_state = nest.flatten(states)
                flat_new_state = nest.flatten(new_states)
                for state, new_state in zip(flat_state, flat_new_state):
                    if isinstance(new_state, tensor_lib.Tensor):
                        new_state.set_shape(state.shape)
                flat_output = nest.flatten(output)
                output_ta_t = tuple((ta.write(time, out) for ta, out in zip(output_ta_t, flat_output)))
                new_states = nest.pack_sequence_as(initial_states, flat_new_state)
                return (time + 1, output_ta_t) + tuple(new_states)
            final_outputs = while_loop.while_loop(body=_step, loop_vars=(time, output_ta) + states, **while_loop_kwargs)
            new_states = final_outputs[2:]
        output_ta = final_outputs[1]
        outputs = tuple((o.stack() for o in output_ta))
        last_output = tuple((o[-1] for o in outputs))
        outputs = nest.pack_sequence_as(output_time_zero, outputs)
        last_output = nest.pack_sequence_as(output_time_zero, last_output)

    def set_shape(output_):
        if isinstance(output_, tensor_lib.Tensor):
            shape = output_.shape.as_list()
            shape[0] = time_steps
            shape[1] = batch
            output_.set_shape(shape)
        return output_
    outputs = nest.map_structure(set_shape, outputs)
    if not time_major:
        outputs = nest.map_structure(swap_batch_timestep, outputs)
    return (last_output, outputs, new_states)