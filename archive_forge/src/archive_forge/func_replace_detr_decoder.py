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
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
from PIL import Image
from torch import Tensor, nn
from transformers.models.maskformer.feature_extraction_maskformer import MaskFormerImageProcessor
from transformers.models.maskformer.modeling_maskformer import (
from transformers.utils import logging
def replace_detr_decoder(self, dst_state_dict: StateDict, src_state_dict: StateDict):
    dst_prefix: str = 'transformer_module.decoder'
    src_prefix: str = 'sem_seg_head.predictor.transformer.decoder'
    renamed_keys = self.rename_keys_in_detr_decoder(dst_state_dict, src_state_dict)
    renamed_keys.extend([(f'{src_prefix}.norm.weight', f'{dst_prefix}.layernorm.weight'), (f'{src_prefix}.norm.bias', f'{dst_prefix}.layernorm.bias')])
    self.pop_all(renamed_keys, dst_state_dict, src_state_dict)
    self.replace_q_k_v_in_detr_decoder(dst_state_dict, src_state_dict)