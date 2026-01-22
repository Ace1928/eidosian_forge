from typing import Dict, List, Optional, Tuple
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.ops import boxes as box_ops, Conv2dNormActivation
from . import _utils as det_utils
from .anchor_utils import AnchorGenerator  # noqa: 401
from .image_list import ImageList
def assign_targets_to_anchors(self, anchors: List[Tensor], targets: List[Dict[str, Tensor]]) -> Tuple[List[Tensor], List[Tensor]]:
    labels = []
    matched_gt_boxes = []
    for anchors_per_image, targets_per_image in zip(anchors, targets):
        gt_boxes = targets_per_image['boxes']
        if gt_boxes.numel() == 0:
            device = anchors_per_image.device
            matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
            labels_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)
        else:
            match_quality_matrix = self.box_similarity(gt_boxes, anchors_per_image)
            matched_idxs = self.proposal_matcher(match_quality_matrix)
            matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]
            labels_per_image = matched_idxs >= 0
            labels_per_image = labels_per_image.to(dtype=torch.float32)
            bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indices] = 0.0
            inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
            labels_per_image[inds_to_discard] = -1.0
        labels.append(labels_per_image)
        matched_gt_boxes.append(matched_gt_boxes_per_image)
    return (labels, matched_gt_boxes)