from typing import Callable, Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring, overwrite_call_docstring
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_roformer import RoFormerConfig
def rotate_layer(layer, sin_pos, cos_pos):
    rotate_half_layer = jnp.stack([-layer[..., 1::2], layer[..., ::2]], axis=-1).reshape(layer.shape)
    rotary_matrix_cos = jnp.einsum('bslh,...sh->bslh', layer, cos_pos)
    rotary_matrix_sin = jnp.einsum('bslh,...sh->bslh', rotate_half_layer, sin_pos)
    return rotary_matrix_cos + rotary_matrix_sin