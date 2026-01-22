from __future__ import annotations
import collections.abc
import math
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import ACT2FN
from ...modeling_tf_utils import (
from ...tf_utils import shape_list
from ...utils import (
from .configuration_swin import SwinConfig
def compute_output_shape(self, input_shape: Iterable[int]) -> tf.TensorShape:
    input_shape = tf.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_last':
        shape = tf.TensorShape([input_shape[0], self.output_size[0], input_shape[2]])
    else:
        shape = tf.TensorShape([input_shape[0], input_shape[1], self.output_size[0]])
    return shape