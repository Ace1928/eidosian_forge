from functools import partial
from typing import Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_llama import LlamaConfig
def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict=None) -> FrozenDict:
    input_ids = jnp.zeros(input_shape, dtype='i4')
    attention_mask = jnp.ones_like(input_ids)
    position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
    params_rng, dropout_rng = jax.random.split(rng)
    rngs = {'params': params_rng, 'dropout': dropout_rng}
    random_params = self.module.init(rngs, input_ids, attention_mask, position_ids, return_dict=False)['params']
    if params is not None:
        random_params = flatten_dict(unfreeze(random_params))
        params = flatten_dict(unfreeze(params))
        for missing_key in self._missing_keys:
            params[missing_key] = random_params[missing_key]
        self._missing_keys = set()
        return freeze(unflatten_dict(params))
    else:
        return random_params