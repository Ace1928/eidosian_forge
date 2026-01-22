import collections
import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_layoutlmv3 import LayoutLMv3Config
def init_visual_bbox(self, image_size=(14, 14), max_len=1000):
    """
        Create the bounding boxes for the visual (patch) tokens.
        """
    visual_bbox_x = torch.div(torch.arange(0, max_len * (image_size[1] + 1), max_len), image_size[1], rounding_mode='trunc')
    visual_bbox_y = torch.div(torch.arange(0, max_len * (image_size[0] + 1), max_len), image_size[0], rounding_mode='trunc')
    visual_bbox = torch.stack([visual_bbox_x[:-1].repeat(image_size[0], 1), visual_bbox_y[:-1].repeat(image_size[1], 1).transpose(0, 1), visual_bbox_x[1:].repeat(image_size[0], 1), visual_bbox_y[1:].repeat(image_size[1], 1).transpose(0, 1)], dim=-1).view(-1, 4)
    cls_token_box = torch.tensor([[0 + 1, 0 + 1, max_len - 1, max_len - 1]])
    self.visual_bbox = torch.cat([cls_token_box, visual_bbox], dim=0)