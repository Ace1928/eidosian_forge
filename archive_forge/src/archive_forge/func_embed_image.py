from __future__ import annotations
import collections
import math
from typing import List, Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .configuration_layoutlmv3 import LayoutLMv3Config
def embed_image(self, pixel_values: tf.Tensor) -> tf.Tensor:
    embeddings = self.patch_embed(pixel_values)
    batch_size = tf.shape(embeddings)[0]
    cls_tokens = tf.tile(self.cls_token, [batch_size, 1, 1])
    embeddings = tf.concat([cls_tokens, embeddings], axis=1)
    if getattr(self, 'pos_embed', None) is not None:
        embeddings += self.pos_embed
    embeddings = self.norm(embeddings)
    return embeddings