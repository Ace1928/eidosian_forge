import argparse
import collections
import jax
import jax.numpy as jnp
import torch
import torch.nn as nn
from clip.model import CLIP
from flax.training import checkpoints
from huggingface_hub import Repository
from transformers import (
def copy_class_box_heads(hf_model, flax_params):
    pt_params = hf_model.state_dict()
    new_params = {}
    flax_class_params = flatten_nested_dict(flax_params['class_head'])
    for flax_key, v in flax_class_params.items():
        torch_key = flax_key.replace('/', '.')
        torch_key = torch_key.replace('.kernel', '.weight')
        torch_key = torch_key.replace('Dense_0', 'dense0')
        torch_key = 'class_head.' + torch_key
        if 'weight' in torch_key and v.ndim == 2:
            v = v.T
        new_params[torch_key] = nn.Parameter(torch.from_numpy(v))
    flax_box_params = flatten_nested_dict(flax_params['obj_box_head'])
    for flax_key, v in flax_box_params.items():
        torch_key = flax_key.replace('/', '.')
        torch_key = torch_key.replace('.kernel', '.weight')
        torch_key = torch_key.replace('_', '').lower()
        torch_key = 'box_head.' + torch_key
        if 'weight' in torch_key and v.ndim == 2:
            v = v.T
        new_params[torch_key] = nn.Parameter(torch.from_numpy(v))
    for name, param in new_params.items():
        if name in pt_params.keys():
            pt_params[name].copy_(param)