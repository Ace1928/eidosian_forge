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
def replace_text_mapper(self, dst_state_dict: StateDict, src_state_dict: StateDict):
    dst_prefix: str = 'text_mapper.text_encoder'
    src_prefix: str = 'text_encoder'
    self.replace_text_projector(dst_state_dict, src_state_dict)

    def rename_keys_for_weight_bias(src_prefix: str, dst_prefix: str):
        return [(f'{src_prefix}.weight', f'{dst_prefix}.weight'), (f'{src_prefix}.bias', f'{dst_prefix}.bias')]

    def rename_keys_for_attn(src_prefix: str, dst_prefix: str):
        attn_keys = [(f'{src_prefix}.in_proj_bias', f'{dst_prefix}.in_proj_bias'), (f'{src_prefix}.in_proj_weight', f'{dst_prefix}.in_proj_weight')]
        attn_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.out_proj', f'{dst_prefix}.out_proj'))
        return attn_keys

    def rename_keys_for_layer(src_prefix: str, dst_prefix: str):
        resblock_keys = []
        resblock_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.mlp.c_fc', f'{dst_prefix}.mlp.fc1'))
        resblock_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.mlp.c_proj', f'{dst_prefix}.mlp.fc2'))
        resblock_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.ln_1', f'{dst_prefix}.layer_norm1'))
        resblock_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.ln_2', f'{dst_prefix}.layer_norm2'))
        resblock_keys.extend(rename_keys_for_attn(f'{src_prefix}.attn', f'{dst_prefix}.self_attn'))
        return resblock_keys
    renamed_keys = [('prompt_ctx.weight', 'text_mapper.prompt_ctx.weight')]
    renamed_keys.extend([(f'{src_prefix}.positional_embedding', f'{dst_prefix}.positional_embedding'), (f'{src_prefix}.token_embedding.weight', f'{dst_prefix}.token_embedding.weight')])
    renamed_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.ln_final', f'{dst_prefix}.ln_final'))
    for i in range(self.config.text_encoder_config['text_encoder_num_layers']):
        renamed_keys.extend(rename_keys_for_layer(f'{src_prefix}.transformer.resblocks.{i}', f'{dst_prefix}.transformer.layers.{i}'))
    self.pop_all(renamed_keys, dst_state_dict, src_state_dict)