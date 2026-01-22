import gc
import json
import os
import re
import warnings
from functools import partial
from pickle import UnpicklingError
from typing import Any, Dict, Optional, Set, Tuple, Union
import flax.linen as nn
import jax
import jax.numpy as jnp
import msgpack.exceptions
from flax.core.frozen_dict import FrozenDict, unfreeze
from flax.serialization import from_bytes, to_bytes
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.random import PRNGKey
from .configuration_utils import PretrainedConfig
from .dynamic_module_utils import custom_object_save
from .generation import FlaxGenerationMixin, GenerationConfig
from .modeling_flax_pytorch_utils import load_pytorch_checkpoint_in_flax_state_dict
from .utils import (
from .utils.hub import convert_file_size_to_int, get_checkpoint_shard_files
from .utils.import_utils import is_safetensors_available
@classmethod
def load_flax_sharded_weights(cls, shard_files):
    """
        This is the same as [`flax.serialization.from_bytes`]
        (https:lax.readthedocs.io/en/latest/_modules/flax/serialization.html#from_bytes) but for a sharded checkpoint.

        This load is performed efficiently: each checkpoint shard is loaded one by one in RAM and deleted after being
        loaded in the model.

        Args:
            shard_files (`List[str]`:
                The list of shard files to load.

        Returns:
            `Dict`: A nested dictionary of the model parameters, in the expected format for flax models : `{'model':
            {'params': {'...'}}}`.
        """
    state_sharded_dict = {}
    for shard_file in shard_files:
        try:
            with open(shard_file, 'rb') as state_f:
                state = from_bytes(cls, state_f.read())
        except (UnpicklingError, msgpack.exceptions.ExtraData) as e:
            with open(shard_file) as f:
                if f.read().startswith('version'):
                    raise OSError('You seem to have cloned a repository without having git-lfs installed. Please install git-lfs and run `git lfs install` followed by `git lfs pull` in the folder you cloned.')
                else:
                    raise ValueError from e
        except (UnicodeDecodeError, ValueError):
            raise EnvironmentError(f'Unable to convert {shard_file} to Flax deserializable object. ')
        state = flatten_dict(state, sep='/')
        state_sharded_dict.update(state)
        del state
        gc.collect()
    return unflatten_dict(state_sharded_dict, sep='/')