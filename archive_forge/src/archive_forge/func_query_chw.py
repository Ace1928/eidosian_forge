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
def query_chw(flat_inputs: List[Any]) -> Tuple[int, int, int]:
    chws = {tuple(get_dimensions(inpt)) for inpt in flat_inputs if check_type(inpt, (is_pure_tensor, tv_tensors.Image, PIL.Image.Image, tv_tensors.Video))}
    if not chws:
        raise TypeError('No image or video was found in the sample')
    elif len(chws) > 1:
        raise ValueError(f'Found multiple CxHxW dimensions in the sample: {sequence_to_str(sorted(chws))}')
    c, h, w = chws.pop()
    return (c, h, w)