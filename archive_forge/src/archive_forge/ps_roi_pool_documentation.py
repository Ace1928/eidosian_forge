import torch
import torch.fx
from torch import nn, Tensor
from torch.nn.modules.utils import _pair
from torchvision.extension import _assert_has_ops
from ..utils import _log_api_usage_once
from ._utils import check_roi_boxes_shape, convert_boxes_to_roi_format

    See :func:`ps_roi_pool`.
    