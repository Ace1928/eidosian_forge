from logging import getLogger
from typing import Optional, Union
import torch
from torch import nn
from transformers.pytorch_utils import Conv1D
from .constants import BLOCK_PATTERNS, SEQLEN_KEYS_TRANFORMERS
def get_seqlen(model: nn.Module):
    if hasattr(model, 'config'):
        model_config = model.config.to_dict()
        if any((k in model_config for k in SEQLEN_KEYS_TRANFORMERS)):
            for key in SEQLEN_KEYS_TRANFORMERS:
                if key in model_config:
                    return model_config[key]
    logger.info("We couldn't get the model sequence length. Setting it to 2048. You can overwrite this value by passing `model_seqlen` in` GPTQQuantizer`")
    return 2048