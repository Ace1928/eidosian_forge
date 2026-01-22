import argparse
import collections
import os
import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax.training import checkpoints
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import (
from transformers.utils import logging
def rename_and_reshape_key(dct, old, new, config):
    val = dct.pop(old)
    if ('out_proj' in new or 'v_proj' in new or 'k_proj' in new or ('q_proj' in new)) and 'vision' in new:
        val = val.reshape(-1, config.vision_config.hidden_size)
    if ('out_proj' in new or 'v_proj' in new or 'k_proj' in new or ('q_proj' in new)) and 'text' in new:
        val = val.reshape(-1, config.text_config.hidden_size)
    if 'patch_embedding' in new:
        print('Reshaping patch embedding... for', new)
        val = val.transpose(3, 2, 0, 1)
    elif new.endswith('weight') and 'position_embedding' not in new and ('token_embedding' not in new):
        val = val.T
    if new.endswith('bias'):
        val = val.reshape(-1)
    dct[new] = torch.from_numpy(np.array(val))