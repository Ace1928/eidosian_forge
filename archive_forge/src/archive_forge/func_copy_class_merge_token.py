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
def copy_class_merge_token(hf_model, flax_params):
    flax_class_token_params = flatten_nested_dict(flax_params['backbone']['merged_class_token'])
    weight = torch.from_numpy(flax_class_token_params['scale'])
    bias = torch.from_numpy(flax_class_token_params['bias'])
    hf_model.layer_norm.weight = nn.Parameter(weight)
    hf_model.layer_norm.bias = nn.Parameter(bias)