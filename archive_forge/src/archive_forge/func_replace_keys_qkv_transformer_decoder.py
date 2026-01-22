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
def replace_keys_qkv_transformer_decoder(self, dst_state_dict: StateDict, src_state_dict: StateDict):
    dst_prefix: str = 'transformer_module.decoder.layers'
    src_prefix: str = 'sem_seg_head.predictor'
    for i in range(self.config.decoder_layers - 1):
        in_proj_weight = src_state_dict.pop(f'{src_prefix}.transformer_self_attention_layers.{i}.self_attn.in_proj_weight')
        in_proj_bias = src_state_dict.pop(f'{src_prefix}.transformer_self_attention_layers.{i}.self_attn.in_proj_bias')
        dst_state_dict[f'{dst_prefix}.{i}.self_attn.self_attn.q_proj.weight'] = in_proj_weight[:256, :]
        dst_state_dict[f'{dst_prefix}.{i}.self_attn.self_attn.q_proj.bias'] = in_proj_bias[:256]
        dst_state_dict[f'{dst_prefix}.{i}.self_attn.self_attn.k_proj.weight'] = in_proj_weight[256:512, :]
        dst_state_dict[f'{dst_prefix}.{i}.self_attn.self_attn.k_proj.bias'] = in_proj_bias[256:512]
        dst_state_dict[f'{dst_prefix}.{i}.self_attn.self_attn.v_proj.weight'] = in_proj_weight[-256:, :]
        dst_state_dict[f'{dst_prefix}.{i}.self_attn.self_attn.v_proj.bias'] = in_proj_bias[-256:]