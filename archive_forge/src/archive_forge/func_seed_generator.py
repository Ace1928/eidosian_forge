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
@property
def seed_generator(self):
    warnings.warn('`seed_generator` is deprecated and will be removed in a future version.', UserWarning)
    if self._seed_generator is None:
        self._seed_generator = tf.random.Generator.from_non_deterministic_state()
    return self._seed_generator