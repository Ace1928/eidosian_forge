import warnings
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from ...ops import boxes as box_ops
from ...transforms._presets import ObjectDetection
from ...utils import _log_api_usage_once
from .._api import register_model, Weights, WeightsEnum
from .._meta import _COCO_CATEGORIES
from .._utils import _ovewrite_value_param, handle_legacy_interface
from ..vgg import VGG, vgg16, VGG16_Weights
from . import _utils as det_utils
from .anchor_utils import DefaultBoxGenerator
from .backbone_utils import _validate_trainable_layers
from .transform import GeneralizedRCNNTransform
def postprocess_detections(self, head_outputs: Dict[str, Tensor], image_anchors: List[Tensor], image_shapes: List[Tuple[int, int]]) -> List[Dict[str, Tensor]]:
    bbox_regression = head_outputs['bbox_regression']
    pred_scores = F.softmax(head_outputs['cls_logits'], dim=-1)
    num_classes = pred_scores.size(-1)
    device = pred_scores.device
    detections: List[Dict[str, Tensor]] = []
    for boxes, scores, anchors, image_shape in zip(bbox_regression, pred_scores, image_anchors, image_shapes):
        boxes = self.box_coder.decode_single(boxes, anchors)
        boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
        image_boxes = []
        image_scores = []
        image_labels = []
        for label in range(1, num_classes):
            score = scores[:, label]
            keep_idxs = score > self.score_thresh
            score = score[keep_idxs]
            box = boxes[keep_idxs]
            num_topk = det_utils._topk_min(score, self.topk_candidates, 0)
            score, idxs = score.topk(num_topk)
            box = box[idxs]
            image_boxes.append(box)
            image_scores.append(score)
            image_labels.append(torch.full_like(score, fill_value=label, dtype=torch.int64, device=device))
        image_boxes = torch.cat(image_boxes, dim=0)
        image_scores = torch.cat(image_scores, dim=0)
        image_labels = torch.cat(image_labels, dim=0)
        keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
        keep = keep[:self.detections_per_img]
        detections.append({'boxes': image_boxes[keep], 'scores': image_scores[keep], 'labels': image_labels[keep]})
    return detections