import os
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, Iterator, List, Set, Tuple
import requests
import torch
import torchvision.transforms as T
from PIL import Image
from torch import Tensor, nn
from transformers import CLIPTokenizer, DinatConfig, SwinConfig
from transformers.models.oneformer.image_processing_oneformer import OneFormerImageProcessor
from transformers.models.oneformer.modeling_oneformer import (
from transformers.models.oneformer.processing_oneformer import OneFormerProcessor
from transformers.utils import logging
def rename_keys_for_layer(src_prefix: str, dst_prefix: str):
    resblock_keys = []
    resblock_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.mlp.c_fc', f'{dst_prefix}.mlp.fc1'))
    resblock_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.mlp.c_proj', f'{dst_prefix}.mlp.fc2'))
    resblock_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.ln_1', f'{dst_prefix}.layer_norm1'))
    resblock_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.ln_2', f'{dst_prefix}.layer_norm2'))
    resblock_keys.extend(rename_keys_for_attn(f'{src_prefix}.attn', f'{dst_prefix}.self_attn'))
    return resblock_keys