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
def replace_universal_segmentation_module(self, dst_state_dict: StateDict, src_state_dict: StateDict):
    dst_prefix: str = ''
    src_prefix: str = 'sem_seg_head.predictor'
    renamed_keys = [(f'{src_prefix}.class_embed.weight', f'{dst_prefix}class_predictor.weight'), (f'{src_prefix}.class_embed.bias', f'{dst_prefix}class_predictor.bias')]
    logger.info(f'Replacing keys {pformat(renamed_keys)}')
    self.pop_all(renamed_keys, dst_state_dict, src_state_dict)