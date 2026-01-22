import os
import warnings
from typing import Optional
import torch
from huggingface_hub import file_exists, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from safetensors.torch import load_file as safe_load_file
from .other import (
from .peft_types import PeftType
def get_embedding_layer_name(model, layer, is_embedding_in_target_modules):
    """Get the name of the embedding module for a given layer."""
    for name, module in model.named_modules():
        if not is_embedding_in_target_modules and module == layer or module == getattr(layer, 'base_layer', None):
            return name
    return None