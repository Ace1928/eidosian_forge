import copy
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn
from torch.cuda.amp import autocast
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import load_backbone
from .configuration_oneformer import OneFormerConfig
def get_loss_dict(self, masks_queries_logits: Tensor, class_queries_logits: Tensor, contrastive_queries_logits: Tensor, mask_labels: Tensor, class_labels: Tensor, text_queries: Tensor, auxiliary_predictions: Dict[str, Tensor], calculate_contrastive_loss: bool) -> Dict[str, Tensor]:
    loss_dict: Dict[str, Tensor] = self.criterion(masks_queries_logits=masks_queries_logits, class_queries_logits=class_queries_logits, contrastive_queries_logits=contrastive_queries_logits, mask_labels=mask_labels, class_labels=class_labels, text_queries=text_queries, auxiliary_predictions=auxiliary_predictions, calculate_contrastive_loss=calculate_contrastive_loss)
    for key, weight in self.weight_dict.items():
        for loss_key, loss in loss_dict.items():
            if key in loss_key:
                loss *= weight
    return loss_dict