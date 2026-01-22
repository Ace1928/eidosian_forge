from typing import List, Union
import torch
import torch._dynamo
import torch.fx
from torch import nn, Tensor
from torch.jit.annotations import BroadcastingList2
from torch.nn.modules.utils import _pair
from torchvision.extension import _assert_has_ops, _has_ops
from ..utils import _log_api_usage_once
from ._utils import check_roi_boxes_shape, convert_boxes_to_roi_format
def masked_index(y, x):
    if ymask is not None:
        assert xmask is not None
        y = torch.where(ymask[:, None, :], y, 0)
        x = torch.where(xmask[:, None, :], x, 0)
    return input[roi_batch_ind[:, None, None, None, None, None], torch.arange(channels, device=input.device)[None, :, None, None, None, None], y[:, None, :, None, :, None], x[:, None, None, :, None, :]]