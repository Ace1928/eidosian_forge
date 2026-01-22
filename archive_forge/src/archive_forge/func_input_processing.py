from __future__ import annotations
import functools
import gc
import inspect
import json
import os
import pickle
import re
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
import h5py
import numpy as np
import tensorflow as tf
from packaging.version import parse
from . import DataCollatorWithPadding, DefaultDataCollator
from .activations_tf import get_tf_activation
from .configuration_utils import PretrainedConfig
from .dynamic_module_utils import custom_object_save
from .generation import GenerationConfig, TFGenerationMixin
from .tf_utils import (
from .utils import (
from .utils.hub import convert_file_size_to_int, get_checkpoint_shard_files
def input_processing(func, config, **kwargs):
    """
    Process the input of each TensorFlow model including the booleans. In case of a list of symbolic inputs, each input
    has to be named accordingly to the parameters name, i.e. `input_ids = keras.Input(shape=(128,), dtype='int32',
    name="input_ids")` otherwise the order of the tensors will not be guaranteed during the training.

    Args:
        func (`callable`):
            The callable function of the TensorFlow model.
        config ([`PretrainedConfig`]):
            The config of the running model.
        **kwargs:
            The inputs of the model.

    Returns:
        Two lists, one for the missing layers, and another one for the unexpected layers.
    """
    signature = dict(inspect.signature(func).parameters)
    has_kwargs = bool(signature.pop('kwargs', None))
    signature.pop('self', None)
    parameter_names = list(signature.keys())
    main_input_name = parameter_names[0]
    main_input = kwargs.pop(main_input_name, None)
    output = {}
    allowed_types = (tf.Tensor, bool, int, ModelOutput, tuple, list, dict, np.ndarray)
    if 'inputs' in kwargs['kwargs_call']:
        warnings.warn('The `inputs` argument is deprecated and will be removed in a future version, use `input_ids` instead.', FutureWarning)
        output['input_ids'] = kwargs['kwargs_call'].pop('inputs')
    if 'decoder_cached_states' in kwargs['kwargs_call']:
        warnings.warn('The `decoder_cached_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.', FutureWarning)
        output['past_key_values'] = kwargs['kwargs_call'].pop('decoder_cached_states')
    if 'past' in kwargs['kwargs_call'] and 'past_key_values' in parameter_names:
        warnings.warn('The `past` argument is deprecated and will be removed in a future version, use `past_key_values` instead.', FutureWarning)
        kwargs['past_key_values'] = kwargs['kwargs_call'].pop('past')
    elif 'past_key_values' in kwargs['kwargs_call'] and 'past' in parameter_names:
        kwargs['past'] = kwargs['kwargs_call'].pop('past_key_values')
    if has_kwargs:
        output['kwargs'] = kwargs.pop('kwargs_call', {})
    else:
        if len(kwargs['kwargs_call']) > 0:
            raise ValueError(f'The following keyword arguments are not supported by this model: {list(kwargs['kwargs_call'].keys())}.')
        kwargs.pop('kwargs_call')
    for k, v in kwargs.items():
        if isinstance(v, allowed_types) or tf.is_tensor(v) or v is None:
            output[k] = v
        else:
            raise ValueError(f'Data of type {type(v)} is not allowed only {allowed_types} is accepted for {k}.')
    if isinstance(main_input, (tuple, list)):
        for i, input in enumerate(main_input):
            if is_tf_symbolic_tensor(input):
                tensor_name = input.name.split(':')[0]
                if tensor_name in parameter_names:
                    output[tensor_name] = input
                else:
                    output[parameter_names[i]] = input
            elif isinstance(input, allowed_types) or input is None:
                output[parameter_names[i]] = input
            else:
                raise ValueError(f'Data of type {type(input)} is not allowed only {allowed_types} is accepted for {parameter_names[i]}.')
    elif isinstance(main_input, Mapping):
        if 'inputs' in main_input:
            warnings.warn('The `inputs` argument is deprecated and will be removed in a future version, use `input_ids` instead.', FutureWarning)
            output['input_ids'] = main_input.pop('inputs')
        if 'decoder_cached_states' in main_input:
            warnings.warn('The `decoder_cached_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.', FutureWarning)
            output['past_key_values'] = main_input.pop('decoder_cached_states')
        for k, v in dict(main_input).items():
            if isinstance(v, allowed_types) or v is None:
                output[k] = v
            elif k not in parameter_names and 'args' not in parameter_names:
                logger.warning(f'The parameter {k} does not belongs to the parameter list {parameter_names} and will be ignored.')
                continue
            else:
                raise ValueError(f'Data of type {type(v)} is not allowed only {allowed_types} is accepted for {k}.')
    elif tf.is_tensor(main_input) or main_input is None:
        output[main_input_name] = main_input
    else:
        raise ValueError(f'Data of type {type(main_input)} is not allowed only {allowed_types} is accepted for {main_input_name}.')
    for name in parameter_names:
        if name not in list(output.keys()) and name != 'args':
            output[name] = kwargs.pop(name, signature[name].default)
    if 'args' in output:
        if output['args'] is not None and is_tf_symbolic_tensor(output['args']):
            tensor_name = output['args'].name.split(':')[0]
            output[tensor_name] = output['args']
        else:
            output['input_ids'] = output['args']
        del output['args']
    if 'kwargs' in output:
        del output['kwargs']
    cast_output = {}
    for key, val in output.items():
        if isinstance(val, tf.Tensor) and val.dtype == tf.int64:
            cast_output[key] = tf.cast(val, tf.int32)
        elif isinstance(val, np.ndarray) and val.dtype == np.int64:
            cast_output[key] = val.astype(np.int32)
        else:
            cast_output[key] = val
    output = cast_output
    del cast_output
    if config is not None:
        boolean_dict = {k: v for k, v in output.items() if k in ['return_dict', 'output_attentions', 'output_hidden_states', 'use_cache']}
        output.update(booleans_processing(config=config, **boolean_dict))
    return output