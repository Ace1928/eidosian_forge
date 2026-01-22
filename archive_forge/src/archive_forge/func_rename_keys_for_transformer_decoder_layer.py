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
def rename_keys_for_transformer_decoder_layer(src_prefix: str, dst_prefix: str, idx: int):
    transformer_decoder_layer_keys = []
    transformer_decoder_layer_keys.extend(rename_keys_for_cross_attn_layer(f'{src_prefix}.transformer_cross_attention_layers.{idx}', f'{dst_prefix}.{idx}.cross_attn'))
    transformer_decoder_layer_keys.extend(rename_keys_for_self_attn_layer(f'{src_prefix}.transformer_self_attention_layers.{idx}', f'{dst_prefix}.{idx}.self_attn'))
    transformer_decoder_layer_keys.extend(rename_keys_for_ffn_layer(f'{src_prefix}.transformer_ffn_layers.{idx}', f'{dst_prefix}.{idx}.ffn'))
    return transformer_decoder_layer_keys