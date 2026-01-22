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
def get_label_to_output_name_mapping(self):
    arg_names = list(inspect.signature(self.call).parameters)
    if self._label_to_output_map is not None:
        return self._label_to_output_map
    elif 'start_positions' in arg_names:
        return {'start_positions': 'start_logits', 'end_positions': 'end_logits'}
    elif 'sentence_order_label' in arg_names:
        return {'labels': 'prediction_logits', 'sentence_order_label': 'sop_logits'}
    elif 'next_sentence_label' in arg_names:
        return {'labels': 'prediction_logits', 'next_sentence_label': 'seq_relationship_logits'}
    elif 'mc_labels' in arg_names:
        return {'labels': 'logits', 'mc_labels': 'mc_logits'}
    else:
        return {}