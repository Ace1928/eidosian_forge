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
def replace_text_projector(self, dst_state_dict: StateDict, src_state_dict: StateDict):
    dst_prefix: str = 'text_mapper.text_projector'
    src_prefix: str = 'text_projector'

    def rename_keys_for_weight_bias(src_prefix: str, dst_prefix: str):
        return [(f'{src_prefix}.weight', f'{dst_prefix}.weight'), (f'{src_prefix}.bias', f'{dst_prefix}.bias')]
    renamed_keys = []
    for i in range(self.config.text_encoder_config['text_encoder_proj_layers']):
        renamed_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.layers.{i}', f'{dst_prefix}.{i}.0'))
    self.pop_all(renamed_keys, dst_state_dict, src_state_dict)