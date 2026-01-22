import argparse
import collections
import numpy as np
import torch
from flax import traverse_util
from t5x import checkpoints
from transformers import MT5Config, UMT5EncoderModel, UMT5ForConditionalGeneration
from transformers.utils import logging
def t5x_mlp_lookup(params, i, prefix, split_mlp_wi=False):
    """Returns the MLP parameters of a layer. Does not transpose."""
    if split_mlp_wi:
        wi_0 = params[f'{prefix}/{prefix}/mlp/wi_0/kernel'][:, i, :]
        wi_1 = params[f'{prefix}/{prefix}/mlp/wi_1/kernel'][:, i, :]
        wi = (wi_0, wi_1)
    else:
        wi = params[f'{prefix}/{prefix}/mlp/wi/kernel'][:, i, :]
    wo = params[f'{prefix}/{prefix}/mlp/wo/kernel'][:, i, :]
    return (wi, wo)