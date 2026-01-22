import argparse
import json
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Tuple
import timm
import torch
import torch.nn as nn
from classy_vision.models.regnet import RegNet, RegNetParams, RegNetY32gf, RegNetY64gf, RegNetY128gf
from huggingface_hub import cached_download, hf_hub_url
from torch import Tensor
from vissl.models.model_helpers import get_trunk_forward_outputs
from transformers import AutoImageProcessor, RegNetConfig, RegNetForImageClassification, RegNetModel
from transformers.utils import logging
def manually_copy_vissl_head(from_state_dict, to_state_dict, keys: List[Tuple[str, str]]):
    for from_key, to_key in keys:
        to_state_dict[to_key] = from_state_dict[from_key].clone()
        print(f'Copied key={from_key} to={to_key}')
    return to_state_dict