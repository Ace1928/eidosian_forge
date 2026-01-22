import collections
import hashlib
from functools import wraps
import flask
from .dependencies import (
from .exceptions import (
from ._grouping import (
from ._utils import (
from . import _validate
from .long_callback.managers import BaseLongCallbackManager
from ._callback_context import context_value
def insert_callback(callback_list, callback_map, config_prevent_initial_callbacks, output, outputs_indices, inputs, state, inputs_state_indices, prevent_initial_call, long=None, manager=None, running=None, dynamic_creator=False):
    if prevent_initial_call is None:
        prevent_initial_call = config_prevent_initial_callbacks
    _validate.validate_duplicate_output(output, prevent_initial_call, config_prevent_initial_callbacks)
    callback_id = create_callback_id(output, inputs)
    callback_spec = {'output': callback_id, 'inputs': [c.to_dict() for c in inputs], 'state': [c.to_dict() for c in state], 'clientside_function': None, 'prevent_initial_call': prevent_initial_call is True, 'long': long and {'interval': long['interval']}, 'dynamic_creator': dynamic_creator}
    if running:
        callback_spec['running'] = running
    callback_map[callback_id] = {'inputs': callback_spec['inputs'], 'state': callback_spec['state'], 'outputs_indices': outputs_indices, 'inputs_state_indices': inputs_state_indices, 'long': long, 'output': output, 'raw_inputs': inputs, 'manager': manager, 'allow_dynamic_callbacks': dynamic_creator}
    callback_list.append(callback_spec)
    return callback_id