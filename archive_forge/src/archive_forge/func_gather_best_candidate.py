import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from tensorflow.compiler.tf2xla.python.xla import dynamic_update_slice
from ..modeling_tf_outputs import TFCausalLMOutputWithPast, TFSeq2SeqLMOutput
from ..models.auto import (
from ..tf_utils import shape_list, stable_softmax
from ..utils import ModelOutput, logging
from .configuration_utils import GenerationConfig
from .tf_logits_process import (
def gather_best_candidate(nested, selected_idx_stacked, batch_axis=0):
    """Gathers the slices indexed by selected_idx_stacked from a potentially nested structure of tensors."""

    def gather_fn(tensor):
        gathered_tensor = tf.gather(params=tensor, indices=selected_idx_stacked, axis=batch_axis)
        return gathered_tensor
    return tf.nest.map_structure(gather_fn, nested)