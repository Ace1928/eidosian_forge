from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
import torchvision
from torch import nn, Tensor
from torchvision.ops import boxes as box_ops, roi_align
from . import _utils as det_utils
def select_training_samples(self, proposals, targets):
    self.check_targets(targets)
    if targets is None:
        raise ValueError('targets should not be None')
    dtype = proposals[0].dtype
    device = proposals[0].device
    gt_boxes = [t['boxes'].to(dtype) for t in targets]
    gt_labels = [t['labels'] for t in targets]
    proposals = self.add_gt_proposals(proposals, gt_boxes)
    matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
    sampled_inds = self.subsample(labels)
    matched_gt_boxes = []
    num_images = len(proposals)
    for img_id in range(num_images):
        img_sampled_inds = sampled_inds[img_id]
        proposals[img_id] = proposals[img_id][img_sampled_inds]
        labels[img_id] = labels[img_id][img_sampled_inds]
        matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]
        gt_boxes_in_image = gt_boxes[img_id]
        if gt_boxes_in_image.numel() == 0:
            gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
        matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])
    regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
    return (proposals, matched_idxs, labels, regression_targets)