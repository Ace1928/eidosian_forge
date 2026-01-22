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
def convert_instance_segmentation(self, mask_former: MaskFormerForInstanceSegmentation) -> MaskFormerForInstanceSegmentation:
    dst_state_dict = TrackedStateDict(mask_former.state_dict())
    src_state_dict = self.original_model.state_dict()
    self.replace_instance_segmentation_module(dst_state_dict, src_state_dict)
    mask_former.load_state_dict(dst_state_dict)
    return mask_former