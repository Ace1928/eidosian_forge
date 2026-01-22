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
@functools.wraps(func)
def run_call_with_unpacked_inputs(self, *args, **kwargs):
    kwargs_call = {key: val for key, val in kwargs.items() if key not in dict(original_signature.parameters)}
    fn_args_and_kwargs = {key: val for key, val in kwargs.items() if key not in kwargs_call}
    fn_args_and_kwargs.update({'kwargs_call': kwargs_call})
    fn_args_and_kwargs.update(dict(zip(func.__code__.co_varnames[1:], args)))
    if 'EncoderDecoder' in self.__class__.__name__:
        config = None
    else:
        config = self.config
    unpacked_inputs = input_processing(func, config, **fn_args_and_kwargs)
    return func(self, **unpacked_inputs)