import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.ops import complete_box_iou_loss, distance_box_iou_loss, FrozenBatchNorm2d, generalized_box_iou_loss
def encode_single(self, reference_boxes: Tensor, proposals: Tensor) -> Tensor:
    """
        Encode a set of proposals with respect to some
        reference boxes

        Args:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """
    dtype = reference_boxes.dtype
    device = reference_boxes.device
    weights = torch.as_tensor(self.weights, dtype=dtype, device=device)
    targets = encode_boxes(reference_boxes, proposals, weights)
    return targets