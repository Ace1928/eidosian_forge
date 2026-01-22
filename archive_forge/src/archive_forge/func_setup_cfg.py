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
def setup_cfg(args: Args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_common_config(cfg)
    add_oneformer_config(cfg)
    add_swin_config(cfg)
    add_dinat_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    return cfg