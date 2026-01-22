import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.distributed as dist
from torch import IntTensor, Tensor
from torchmetrics.detection.helpers import _fix_empty_tensors, _input_validator
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import _cumsum
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE, _PYCOCOTOOLS_AVAILABLE, _TORCHVISION_GREATER_EQUAL_0_8
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
def compute_area(inputs: List[Any], iou_type: str='bbox') -> Tensor:
    """Compute area of input depending on the specified iou_type.

    Default output for empty input is :class:`~torch.Tensor`

    """
    import pycocotools.mask as mask_utils
    from torchvision.ops import box_area
    if len(inputs) == 0:
        return Tensor([])
    if iou_type == 'bbox':
        return box_area(torch.stack(inputs))
    if iou_type == 'segm':
        inputs = [{'size': i[0], 'counts': i[1]} for i in inputs]
        return torch.tensor(mask_utils.area(inputs).astype('float'))
    raise Exception(f'IOU type {iou_type} is not supported')