import json
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, Iterator, List, Set, Tuple
import requests
import torch
import torchvision.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from huggingface_hub import hf_hub_download
from PIL import Image
from torch import Tensor, nn
from transformers import (
from transformers.models.mask2former.modeling_mask2former import (
from transformers.utils import logging
def rename_keys_in_masked_attention_decoder(self, dst_state_dict: StateDict, src_state_dict: StateDict):
    dst_prefix: str = 'transformer_module.decoder'
    src_prefix: str = 'sem_seg_head.predictor'
    rename_keys = []
    for i in range(self.config.decoder_layers - 1):
        rename_keys.append((f'{src_prefix}.transformer_self_attention_layers.{i}.self_attn.out_proj.weight', f'{dst_prefix}.layers.{i}.self_attn.out_proj.weight'))
        rename_keys.append((f'{src_prefix}.transformer_self_attention_layers.{i}.self_attn.out_proj.bias', f'{dst_prefix}.layers.{i}.self_attn.out_proj.bias'))
        rename_keys.append((f'{src_prefix}.transformer_self_attention_layers.{i}.norm.weight', f'{dst_prefix}.layers.{i}.self_attn_layer_norm.weight'))
        rename_keys.append((f'{src_prefix}.transformer_self_attention_layers.{i}.norm.bias', f'{dst_prefix}.layers.{i}.self_attn_layer_norm.bias'))
        rename_keys.append((f'{src_prefix}.transformer_cross_attention_layers.{i}.multihead_attn.in_proj_weight', f'{dst_prefix}.layers.{i}.cross_attn.in_proj_weight'))
        rename_keys.append((f'{src_prefix}.transformer_cross_attention_layers.{i}.multihead_attn.in_proj_bias', f'{dst_prefix}.layers.{i}.cross_attn.in_proj_bias'))
        rename_keys.append((f'{src_prefix}.transformer_cross_attention_layers.{i}.multihead_attn.out_proj.weight', f'{dst_prefix}.layers.{i}.cross_attn.out_proj.weight'))
        rename_keys.append((f'{src_prefix}.transformer_cross_attention_layers.{i}.multihead_attn.out_proj.bias', f'{dst_prefix}.layers.{i}.cross_attn.out_proj.bias'))
        rename_keys.append((f'{src_prefix}.transformer_cross_attention_layers.{i}.norm.weight', f'{dst_prefix}.layers.{i}.cross_attn_layer_norm.weight'))
        rename_keys.append((f'{src_prefix}.transformer_cross_attention_layers.{i}.norm.bias', f'{dst_prefix}.layers.{i}.cross_attn_layer_norm.bias'))
        rename_keys.append((f'{src_prefix}.transformer_ffn_layers.{i}.linear1.weight', f'{dst_prefix}.layers.{i}.fc1.weight'))
        rename_keys.append((f'{src_prefix}.transformer_ffn_layers.{i}.linear1.bias', f'{dst_prefix}.layers.{i}.fc1.bias'))
        rename_keys.append((f'{src_prefix}.transformer_ffn_layers.{i}.linear2.weight', f'{dst_prefix}.layers.{i}.fc2.weight'))
        rename_keys.append((f'{src_prefix}.transformer_ffn_layers.{i}.linear2.bias', f'{dst_prefix}.layers.{i}.fc2.bias'))
        rename_keys.append((f'{src_prefix}.transformer_ffn_layers.{i}.norm.weight', f'{dst_prefix}.layers.{i}.final_layer_norm.weight'))
        rename_keys.append((f'{src_prefix}.transformer_ffn_layers.{i}.norm.bias', f'{dst_prefix}.layers.{i}.final_layer_norm.bias'))
    return rename_keys