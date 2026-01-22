import argparse
import json
import os
import tensorstore as ts
import torch
from flax import serialization
from flax.traverse_util import flatten_dict, unflatten_dict
from tensorflow.io import gfile
from transformers.modeling_utils import dtype_byte_size
from transformers.models.switch_transformers.convert_switch_transformers_original_flax_checkpoint_to_pytorch import (
from transformers.utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME
from transformers.utils.hub import convert_file_size_to_int
def rename_base_flax_keys(flax_key_tuple, flax_tensor):
    """
    Post renaming of basic JAX keys to pytorch.
    """
    if flax_key_tuple[-1] == 'kernel' and flax_tensor.ndim == 3:
        flax_key_tuple = flax_key_tuple[:-1] + ('weight',)
        flax_tensor = torch.permute(flax_tensor, (0, 2, 1))
    elif flax_key_tuple[-1] == 'kernel' and '.'.join(flax_key_tuple):
        flax_key_tuple = flax_key_tuple[:-1] + ('weight',)
        flax_tensor = flax_tensor.T
    elif flax_key_tuple[-1] in ['scale', 'embedding']:
        flax_key_tuple = flax_key_tuple[:-1] + ('weight',)
    return (flax_key_tuple, flax_tensor)