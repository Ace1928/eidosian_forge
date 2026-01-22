from __future__ import annotations
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...file_utils import (
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import logging
from .configuration_gptj import GPTJConfig
def get_causal_mask(self, key_length, query_length) -> tf.Tensor:
    return tf.cast(self.lower_triangle_mask[:, :, key_length - query_length:key_length, :key_length], tf.bool)