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
def load_tf_weights_from_safetensors(model, resolved_archive_file, ignore_mismatched_sizes=False, _prefix=None):
    with safe_open(resolved_archive_file, framework='tf') as safetensors_archive:
        mismatched_layers = []
        weight_names = [strip_model_name_and_prefix(w.name, _prefix=_prefix) for w in model.weights]
        loaded_weight_names = list(safetensors_archive.keys())
        missing_layers = list(set(weight_names) - set(loaded_weight_names))
        unexpected_layers = list(set(loaded_weight_names) - set(weight_names))
        for weight in model.weights:
            weight_name = strip_model_name_and_prefix(weight.name, _prefix=_prefix)
            if weight_name in loaded_weight_names:
                weight_value = safetensors_archive.get_tensor(weight_name)
                if K.int_shape(weight) != weight_value.shape:
                    try:
                        weight_value = tf.reshape(weight_value, K.int_shape(weight))
                    except (ValueError, tf.errors.InvalidArgumentError) as e:
                        if ignore_mismatched_sizes:
                            mismatched_layers.append((weight_name, weight_value.shape, K.int_shape(weight)))
                            continue
                        else:
                            raise e
                K.set_value(weight, weight_value)
    return (missing_layers, unexpected_layers, mismatched_layers)