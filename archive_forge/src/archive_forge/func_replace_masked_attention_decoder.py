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
def replace_masked_attention_decoder(self, dst_state_dict: StateDict, src_state_dict: StateDict):
    dst_prefix: str = 'transformer_module.decoder'
    src_prefix: str = 'sem_seg_head.predictor'
    renamed_keys = self.rename_keys_in_masked_attention_decoder(dst_state_dict, src_state_dict)
    renamed_keys.extend([(f'{src_prefix}.decoder_norm.weight', f'{dst_prefix}.layernorm.weight'), (f'{src_prefix}.decoder_norm.bias', f'{dst_prefix}.layernorm.bias')])
    mlp_len = 3
    for i in range(mlp_len):
        renamed_keys.extend([(f'{src_prefix}.mask_embed.layers.{i}.weight', f'{dst_prefix}.mask_predictor.mask_embedder.{i}.0.weight'), (f'{src_prefix}.mask_embed.layers.{i}.bias', f'{dst_prefix}.mask_predictor.mask_embedder.{i}.0.bias')])
    self.pop_all(renamed_keys, dst_state_dict, src_state_dict)