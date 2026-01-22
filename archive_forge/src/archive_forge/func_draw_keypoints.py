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
def draw_keypoints(image: torch.Tensor, keypoints: torch.Tensor, connectivity: Optional[List[Tuple[int, int]]]=None, colors: Optional[Union[str, Tuple[int, int, int]]]=None, radius: int=2, width: int=3) -> torch.Tensor:
    """
    Draws Keypoints on given RGB image.
    The values of the input image should be uint8 between 0 and 255.

    Args:
        image (Tensor): Tensor of shape (3, H, W) and dtype uint8.
        keypoints (Tensor): Tensor of shape (num_instances, K, 2) the K keypoints location for each of the N instances,
            in the format [x, y].
        connectivity (List[Tuple[int, int]]]): A List of tuple where,
            each tuple contains pair of keypoints to be connected.
        colors (str, Tuple): The color can be represented as
            PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
        radius (int): Integer denoting radius of keypoint.
        width (int): Integer denoting width of line connecting keypoints.

    Returns:
        img (Tensor[C, H, W]): Image Tensor of dtype uint8 with keypoints drawn.
    """
    if not torch.jit.is_scripting() and (not torch.jit.is_tracing()):
        _log_api_usage_once(draw_keypoints)
    if not isinstance(image, torch.Tensor):
        raise TypeError(f'The image must be a tensor, got {type(image)}')
    elif image.dtype != torch.uint8:
        raise ValueError(f'The image dtype must be uint8, got {image.dtype}')
    elif image.dim() != 3:
        raise ValueError('Pass individual images, not batches')
    elif image.size()[0] != 3:
        raise ValueError('Pass an RGB image. Other Image formats are not supported')
    if keypoints.ndim != 3:
        raise ValueError('keypoints must be of shape (num_instances, K, 2)')
    ndarr = image.permute(1, 2, 0).cpu().numpy()
    img_to_draw = Image.fromarray(ndarr)
    draw = ImageDraw.Draw(img_to_draw)
    img_kpts = keypoints.to(torch.int64).tolist()
    for kpt_id, kpt_inst in enumerate(img_kpts):
        for inst_id, kpt in enumerate(kpt_inst):
            x1 = kpt[0] - radius
            x2 = kpt[0] + radius
            y1 = kpt[1] - radius
            y2 = kpt[1] + radius
            draw.ellipse([x1, y1, x2, y2], fill=colors, outline=None, width=0)
        if connectivity:
            for connection in connectivity:
                start_pt_x = kpt_inst[connection[0]][0]
                start_pt_y = kpt_inst[connection[0]][1]
                end_pt_x = kpt_inst[connection[1]][0]
                end_pt_y = kpt_inst[connection[1]][1]
                draw.line(((start_pt_x, start_pt_y), (end_pt_x, end_pt_y)), width=width)
    return torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1).to(dtype=torch.uint8)