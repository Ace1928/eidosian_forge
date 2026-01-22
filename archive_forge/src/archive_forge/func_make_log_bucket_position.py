from __future__ import annotations
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_deberta_v2 import DebertaV2Config
def make_log_bucket_position(relative_pos, bucket_size, max_position):
    sign = tf.math.sign(relative_pos)
    mid = bucket_size // 2
    abs_pos = tf.where((relative_pos < mid) & (relative_pos > -mid), mid - 1, tf.math.abs(relative_pos))
    log_pos = tf.math.ceil(tf.cast(tf.math.log(abs_pos / mid), tf.float32) / tf.math.log((max_position - 1) / mid) * (mid - 1)) + mid
    bucket_pos = tf.cast(tf.where(abs_pos <= mid, tf.cast(relative_pos, tf.float32), log_pos * tf.cast(sign, tf.float32)), tf.int32)
    return bucket_pos