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
def draw_segmentation_masks(image: torch.Tensor, masks: torch.Tensor, alpha: float=0.8, colors: Optional[Union[List[Union[str, Tuple[int, int, int]]], str, Tuple[int, int, int]]]=None) -> torch.Tensor:
    """
    Draws segmentation masks on given RGB image.
    The values of the input image should be uint8 between 0 and 255.

    Args:
        image (Tensor): Tensor of shape (3, H, W) and dtype uint8.
        masks (Tensor): Tensor of shape (num_masks, H, W) or (H, W) and dtype bool.
        alpha (float): Float number between 0 and 1 denoting the transparency of the masks.
            0 means full transparency, 1 means no transparency.
        colors (color or list of colors, optional): List containing the colors
            of the masks or single color for all masks. The color can be represented as
            PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
            By default, random colors are generated for each mask.

    Returns:
        img (Tensor[C, H, W]): Image Tensor, with segmentation masks drawn on top.
    """
    if not torch.jit.is_scripting() and (not torch.jit.is_tracing()):
        _log_api_usage_once(draw_segmentation_masks)
    if not isinstance(image, torch.Tensor):
        raise TypeError(f'The image must be a tensor, got {type(image)}')
    elif image.dtype != torch.uint8:
        raise ValueError(f'The image dtype must be uint8, got {image.dtype}')
    elif image.dim() != 3:
        raise ValueError('Pass individual images, not batches')
    elif image.size()[0] != 3:
        raise ValueError('Pass an RGB image. Other Image formats are not supported')
    if masks.ndim == 2:
        masks = masks[None, :, :]
    if masks.ndim != 3:
        raise ValueError('masks must be of shape (H, W) or (batch_size, H, W)')
    if masks.dtype != torch.bool:
        raise ValueError(f'The masks must be of dtype bool. Got {masks.dtype}')
    if masks.shape[-2:] != image.shape[-2:]:
        raise ValueError('The image and the masks must have the same height and width')
    num_masks = masks.size()[0]
    if num_masks == 0:
        warnings.warn("masks doesn't contain any mask. No mask was drawn")
        return image
    out_dtype = torch.uint8
    colors = [torch.tensor(color, dtype=out_dtype, device=image.device) for color in _parse_colors(colors, num_objects=num_masks)]
    img_to_draw = image.detach().clone()
    for mask, color in zip(masks, colors):
        img_to_draw[:, mask] = color[:, None]
    out = image * (1 - alpha) + img_to_draw * alpha
    return out.to(out_dtype)