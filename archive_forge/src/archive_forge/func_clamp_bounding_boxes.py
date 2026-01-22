from typing import List, Optional, Tuple
import PIL.Image
import torch
from torchvision import tv_tensors
from torchvision.transforms import _functional_pil as _FP
from torchvision.tv_tensors import BoundingBoxFormat
from torchvision.utils import _log_api_usage_once
from ._utils import _get_kernel, _register_kernel_internal, is_pure_tensor
def clamp_bounding_boxes(inpt: torch.Tensor, format: Optional[BoundingBoxFormat]=None, canvas_size: Optional[Tuple[int, int]]=None) -> torch.Tensor:
    """[BETA] See :func:`~torchvision.transforms.v2.ClampBoundingBoxes` for details."""
    if not torch.jit.is_scripting():
        _log_api_usage_once(clamp_bounding_boxes)
    if torch.jit.is_scripting() or is_pure_tensor(inpt):
        if format is None or canvas_size is None:
            raise ValueError('For pure tensor inputs, `format` and `canvas_size` has to be passed.')
        return _clamp_bounding_boxes(inpt, format=format, canvas_size=canvas_size)
    elif isinstance(inpt, tv_tensors.BoundingBoxes):
        if format is not None or canvas_size is not None:
            raise ValueError('For bounding box tv_tensor inputs, `format` and `canvas_size` must not be passed.')
        output = _clamp_bounding_boxes(inpt.as_subclass(torch.Tensor), format=inpt.format, canvas_size=inpt.canvas_size)
        return tv_tensors.wrap(output, like=inpt)
    else:
        raise TypeError(f'Input can either be a plain tensor or a bounding box tv_tensor, but got {type(inpt)} instead.')