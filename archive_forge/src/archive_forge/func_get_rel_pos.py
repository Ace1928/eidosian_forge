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
def get_rel_pos(self, hidden_states, query_states=None, relative_pos=None):
    if self.relative_attention and relative_pos is None:
        q = shape_list(query_states)[-2] if query_states is not None else shape_list(hidden_states)[-2]
        relative_pos = build_relative_position(q, shape_list(hidden_states)[-2], bucket_size=self.position_buckets, max_position=self.max_relative_positions)
    return relative_pos