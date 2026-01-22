from __future__ import annotations
import collections.abc
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_groupvit import GroupViTConfig, GroupViTTextConfig, GroupViTVisionConfig
def resize_attention_map(attentions: tf.Tensor, height: int, width: int, align_corners: bool=False) -> tf.Tensor:
    """
    Args:
        attentions (`tf.Tensor`): attention map of shape [batch_size, groups, feat_height*feat_width]
        height (`int`): height of the output attention map
        width (`int`): width of the output attention map
        align_corners (`bool`, *optional*): the `align_corner` argument for `nn.functional.interpolate`.

    Returns:
        `tf.Tensor`: resized attention map of shape [batch_size, groups, height, width]
    """
    scale = (height * width // attentions.shape[2]) ** 0.5
    if height > width:
        feat_width = int(np.round(width / scale))
        feat_height = shape_list(attentions)[2] // feat_width
    else:
        feat_height = int(np.round(height / scale))
        feat_width = shape_list(attentions)[2] // feat_height
    batch_size = shape_list(attentions)[0]
    groups = shape_list(attentions)[1]
    attentions = tf.reshape(attentions, (batch_size, groups, feat_height, feat_width))
    attentions = tf.transpose(attentions, perm=(0, 2, 3, 1))
    if align_corners:
        attentions = tf.compat.v1.image.resize(attentions, size=(height, width), method='bilinear', align_corners=align_corners)
    else:
        attentions = tf.image.resize(attentions, size=(height, width), method='bilinear')
    attentions = tf.transpose(attentions, perm=(0, 3, 1, 2))
    return attentions