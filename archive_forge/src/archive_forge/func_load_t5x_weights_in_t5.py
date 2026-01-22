import argparse
import collections
import numpy as np
import torch
from flax import traverse_util
from t5x import checkpoints
from transformers import MT5Config, UMT5EncoderModel, UMT5ForConditionalGeneration
from transformers.utils import logging
def load_t5x_weights_in_t5(model, config, t5x_checkpoint_path, is_encoder_only, scalable_attention):
    """Replaces the params in model witht the T5X converted params."""
    variables = checkpoints.load_t5x_checkpoint(t5x_checkpoint_path)
    converted = convert_t5x_to_pytorch(variables, num_layers=config.num_layers, is_encoder_only=is_encoder_only, scalable_attention=scalable_attention)
    state_dict = make_state_dict(converted, is_encoder_only)
    model.load_state_dict(state_dict, strict=True)