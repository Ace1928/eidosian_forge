import collections
import math
import pathlib
import warnings
from itertools import repeat
from types import FunctionType
from typing import Any, BinaryIO, List, Optional, Tuple, Union
import numpy as np
import torch
from PIL import Image, ImageColor, ImageDraw, ImageFont
@torch.no_grad()
def flow_to_image(flow: torch.Tensor) -> torch.Tensor:
    """
    Converts a flow to an RGB image.

    Args:
        flow (Tensor): Flow of shape (N, 2, H, W) or (2, H, W) and dtype torch.float.

    Returns:
        img (Tensor): Image Tensor of dtype uint8 where each color corresponds
            to a given flow direction. Shape is (N, 3, H, W) or (3, H, W) depending on the input.
    """
    if flow.dtype != torch.float:
        raise ValueError(f'Flow should be of dtype torch.float, got {flow.dtype}.')
    orig_shape = flow.shape
    if flow.ndim == 3:
        flow = flow[None]
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError(f'Input flow should have shape (2, H, W) or (N, 2, H, W), got {orig_shape}.')
    max_norm = torch.sum(flow ** 2, dim=1).sqrt().max()
    epsilon = torch.finfo(flow.dtype).eps
    normalized_flow = flow / (max_norm + epsilon)
    img = _normalized_flow_to_image(normalized_flow)
    if len(orig_shape) == 3:
        img = img[0]
    return img