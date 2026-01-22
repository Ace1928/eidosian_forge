from __future__ import annotations
import collections.abc
import numbers
from contextlib import suppress
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Type, Union
import PIL.Image
import torch
from torchvision import tv_tensors
from torchvision._utils import sequence_to_str
from torchvision.transforms.transforms import _check_sequence_input, _setup_angle, _setup_size  # noqa: F401
from torchvision.transforms.v2.functional import get_dimensions, get_size, is_pure_tensor
from torchvision.transforms.v2.functional._utils import _FillType, _FillTypeJIT
def query_size(flat_inputs: List[Any]) -> Tuple[int, int]:
    sizes = {tuple(get_size(inpt)) for inpt in flat_inputs if check_type(inpt, (is_pure_tensor, tv_tensors.Image, PIL.Image.Image, tv_tensors.Video, tv_tensors.Mask, tv_tensors.BoundingBoxes))}
    if not sizes:
        raise TypeError('No image, video, mask or bounding box was found in the sample')
    elif len(sizes) > 1:
        raise ValueError(f'Found multiple HxW dimensions in the sample: {sequence_to_str(sorted(sizes))}')
    h, w = sizes.pop()
    return (h, w)