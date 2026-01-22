import math
import numbers
import warnings
from typing import Any, List, Optional, Sequence, Tuple, Union
import PIL.Image
import torch
from torch.nn.functional import grid_sample, interpolate, pad as torch_pad
from torchvision import tv_tensors
from torchvision.transforms import _functional_pil as _FP
from torchvision.transforms._functional_tensor import _pad_symmetric
from torchvision.transforms.functional import (
from torchvision.utils import _log_api_usage_once
from ._meta import _get_size_image_pil, clamp_bounding_boxes, convert_bounding_box_format
from ._utils import _FillTypeJIT, _get_kernel, _register_five_ten_crop_kernel_internal, _register_kernel_internal
def perspective_bounding_boxes(bounding_boxes: torch.Tensor, format: tv_tensors.BoundingBoxFormat, canvas_size: Tuple[int, int], startpoints: Optional[List[List[int]]], endpoints: Optional[List[List[int]]], coefficients: Optional[List[float]]=None) -> torch.Tensor:
    if bounding_boxes.numel() == 0:
        return bounding_boxes
    perspective_coeffs = _perspective_coefficients(startpoints, endpoints, coefficients)
    original_shape = bounding_boxes.shape
    bounding_boxes = convert_bounding_box_format(bounding_boxes, old_format=format, new_format=tv_tensors.BoundingBoxFormat.XYXY).reshape(-1, 4)
    dtype = bounding_boxes.dtype if torch.is_floating_point(bounding_boxes) else torch.float32
    device = bounding_boxes.device
    denom = perspective_coeffs[0] * perspective_coeffs[4] - perspective_coeffs[1] * perspective_coeffs[3]
    if denom == 0:
        raise RuntimeError(f'Provided perspective_coeffs {perspective_coeffs} can not be inverted to transform bounding boxes. Denominator is zero, denom={denom}')
    inv_coeffs = [(perspective_coeffs[4] - perspective_coeffs[5] * perspective_coeffs[7]) / denom, (-perspective_coeffs[1] + perspective_coeffs[2] * perspective_coeffs[7]) / denom, (perspective_coeffs[1] * perspective_coeffs[5] - perspective_coeffs[2] * perspective_coeffs[4]) / denom, (-perspective_coeffs[3] + perspective_coeffs[5] * perspective_coeffs[6]) / denom, (perspective_coeffs[0] - perspective_coeffs[2] * perspective_coeffs[6]) / denom, (-perspective_coeffs[0] * perspective_coeffs[5] + perspective_coeffs[2] * perspective_coeffs[3]) / denom, (-perspective_coeffs[4] * perspective_coeffs[6] + perspective_coeffs[3] * perspective_coeffs[7]) / denom, (-perspective_coeffs[0] * perspective_coeffs[7] + perspective_coeffs[1] * perspective_coeffs[6]) / denom]
    theta1 = torch.tensor([[inv_coeffs[0], inv_coeffs[1], inv_coeffs[2]], [inv_coeffs[3], inv_coeffs[4], inv_coeffs[5]]], dtype=dtype, device=device)
    theta2 = torch.tensor([[inv_coeffs[6], inv_coeffs[7], 1.0], [inv_coeffs[6], inv_coeffs[7], 1.0]], dtype=dtype, device=device)
    points = bounding_boxes[:, [[0, 1], [2, 1], [2, 3], [0, 3]]].reshape(-1, 2)
    points = torch.cat([points, torch.ones(points.shape[0], 1, device=points.device)], dim=-1)
    numer_points = torch.matmul(points, theta1.T)
    denom_points = torch.matmul(points, theta2.T)
    transformed_points = numer_points.div_(denom_points)
    transformed_points = transformed_points.reshape(-1, 4, 2)
    out_bbox_mins, out_bbox_maxs = torch.aminmax(transformed_points, dim=1)
    out_bboxes = clamp_bounding_boxes(torch.cat([out_bbox_mins, out_bbox_maxs], dim=1).to(bounding_boxes.dtype), format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=canvas_size)
    return convert_bounding_box_format(out_bboxes, old_format=tv_tensors.BoundingBoxFormat.XYXY, new_format=format, inplace=True).reshape(original_shape)