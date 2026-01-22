import os
from pickle import UnpicklingError
from typing import Dict, Tuple
import jax
import jax.numpy as jnp
import numpy as np
from flax.serialization import from_bytes
from flax.traverse_util import flatten_dict, unflatten_dict
import transformers
from . import is_safetensors_available, is_torch_available
from .utils import logging
def is_key_or_prefix_key_in_dict(key: Tuple[str]) -> bool:
    """Checks if `key` of `(prefix,) + key` is in random_flax_state_dict"""
    return len(set(random_flax_state_dict) & {key, (model_prefix,) + key}) > 0