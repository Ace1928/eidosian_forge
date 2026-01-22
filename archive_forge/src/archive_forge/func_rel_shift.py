from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_xlnet import XLNetConfig
def rel_shift(self, x, klen=-1):
    """perform relative shift to form the relative attention score."""
    x_size = shape_list(x)
    x = tf.reshape(x, (x_size[1], x_size[0], x_size[2], x_size[3]))
    x = x[1:, ...]
    x = tf.reshape(x, (x_size[0], x_size[1] - 1, x_size[2], x_size[3]))
    x = x[:, 0:klen, :, :]
    return x