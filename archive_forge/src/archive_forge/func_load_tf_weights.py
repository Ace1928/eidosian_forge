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
def load_tf_weights(model, resolved_archive_file, ignore_mismatched_sizes=False, _prefix=None):
    """
    Detect missing and unexpected layers and load the TF weights from the shard file accordingly to their names and
    shapes.

    Args:
        model (`keras.models.Model`):
            The model to load the weights into.
        resolved_archive_file (`str`):
            The location of the H5 file.
        ignore_mismatched_sizes (`bool`, *optional*, defaults to `False`):
            Whether or not to ignore weights with shapes that don't match between the checkpoint of the model.

    Returns:
        Three lists, one for the missing layers, another one for the unexpected layers, and a last one for the
        mismatched layers.
    """
    if resolved_archive_file.endswith('.safetensors'):
        load_function = load_tf_weights_from_safetensors
    else:
        load_function = load_tf_weights_from_h5
    return load_function(model, resolved_archive_file, ignore_mismatched_sizes=ignore_mismatched_sizes, _prefix=_prefix)