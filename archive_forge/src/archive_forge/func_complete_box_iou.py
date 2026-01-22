from typing import Tuple
import torch
import torchvision
from torch import Tensor
from torchvision.extension import _assert_has_ops
from ..utils import _log_api_usage_once
from ._box_convert import _box_cxcywh_to_xyxy, _box_xywh_to_xyxy, _box_xyxy_to_cxcywh, _box_xyxy_to_xywh
from ._utils import _upcast
def complete_box_iou(boxes1: Tensor, boxes2: Tensor, eps: float=1e-07) -> Tensor:
    """
    Return complete intersection-over-union (Jaccard index) between two sets of boxes.
    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
    Args:
        boxes1 (Tensor[N, 4]): first set of boxes
        boxes2 (Tensor[M, 4]): second set of boxes
        eps (float, optional): small number to prevent division by zero. Default: 1e-7
    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise complete IoU values
        for every element in boxes1 and boxes2
    """
    if not torch.jit.is_scripting() and (not torch.jit.is_tracing()):
        _log_api_usage_once(complete_box_iou)
    boxes1 = _upcast(boxes1)
    boxes2 = _upcast(boxes2)
    diou, iou = _box_diou_iou(boxes1, boxes2, eps)
    w_pred = boxes1[:, None, 2] - boxes1[:, None, 0]
    h_pred = boxes1[:, None, 3] - boxes1[:, None, 1]
    w_gt = boxes2[:, 2] - boxes2[:, 0]
    h_gt = boxes2[:, 3] - boxes2[:, 1]
    v = 4 / torch.pi ** 2 * torch.pow(torch.atan(w_pred / h_pred) - torch.atan(w_gt / h_gt), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return diou - alpha * v