import argparse
import collections
import numpy as np
import torch
from flax import traverse_util
from t5x import checkpoints
from transformers import MT5Config, UMT5EncoderModel, UMT5ForConditionalGeneration
from transformers.utils import logging
def t5x_relpos_bias_lookup(params, i, prefix):
    """Returns the Relative Position Bias parameters of a layer. Does not transpose."""
    return params[f'{prefix}/{prefix}/relpos_bias/rel_embedding'][:, i, :]