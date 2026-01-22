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
def load_pytorch_checkpoint_in_flax_state_dict(flax_model, pytorch_checkpoint_path, is_sharded, allow_missing_keys=False):
    """Load pytorch checkpoints in a flax model"""
    if not is_sharded:
        pt_path = os.path.abspath(pytorch_checkpoint_path)
        logger.info(f'Loading PyTorch weights from {pt_path}')
        if pt_path.endswith('.safetensors'):
            pt_state_dict = {}
            with safe_open(pt_path, framework='flax') as f:
                for k in f.keys():
                    pt_state_dict[k] = f.get_tensor(k)
        else:
            try:
                import torch
                from .pytorch_utils import is_torch_greater_or_equal_than_1_13
            except (ImportError, ModuleNotFoundError):
                logger.error('Loading a PyTorch model in Flax, requires both PyTorch and Flax to be installed. Please see https://pytorch.org/ and https://flax.readthedocs.io/en/latest/installation.html for installation instructions.')
                raise
            weights_only_kwarg = {'weights_only': True} if is_torch_greater_or_equal_than_1_13 else {}
            pt_state_dict = torch.load(pt_path, map_location='cpu', **weights_only_kwarg)
            logger.info(f'PyTorch checkpoint contains {sum((t.numel() for t in pt_state_dict.values())):,} parameters.')
        flax_state_dict = convert_pytorch_state_dict_to_flax(pt_state_dict, flax_model)
    else:
        flax_state_dict = convert_pytorch_sharded_state_dict_to_flax(pytorch_checkpoint_path, flax_model)
    return flax_state_dict