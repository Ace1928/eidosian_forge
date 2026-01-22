import os
import warnings
from typing import Optional
import torch
from huggingface_hub import file_exists, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from safetensors.torch import load_file as safe_load_file
from .other import (
from .peft_types import PeftType
def has_valid_embedding_base_layer(layer):
    """Check if the layer has an embedding base layer"""
    return hasattr(layer, 'base_layer') and isinstance(layer.base_layer, (torch.nn.Linear, torch.nn.Embedding))