import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import warnings
from collections import OrderedDict
from functools import lru_cache
from itertools import chain
from types import ModuleType
from typing import Any, Tuple, Union
from packaging import version
from . import logging
@lru_cache()
def is_torch_fp16_available_on_device(device):
    if not is_torch_available():
        return False
    import torch
    try:
        x = torch.zeros(2, 2, dtype=torch.float16).to(device)
        _ = x @ x
        batch, sentence_length, embedding_dim = (3, 4, 5)
        embedding = torch.randn(batch, sentence_length, embedding_dim, dtype=torch.float16, device=device)
        layer_norm = torch.nn.LayerNorm(embedding_dim, dtype=torch.float16, device=device)
        _ = layer_norm(embedding)
    except:
        return False
    return True