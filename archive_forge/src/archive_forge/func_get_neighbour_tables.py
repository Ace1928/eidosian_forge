import functools
import math
from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from torch.nn.functional import conv2d, conv3d, pad, unfold
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.imports import _SCIPY_AVAILABLE
@functools.lru_cache
def get_neighbour_tables(spacing: Union[Tuple[int, int], Tuple[int, int, int]], device: Optional[torch.device]=None) -> Tuple[Tensor, Tensor]:
    """Create a table that maps neighbour codes to the contour length or surface area of the corresponding contour.

    Args:
        spacing: The spacing between pixels along each spatial dimension.
        device: The device on which the table should be created.

    Returns:
        A tuple containing as its first element the table that maps neighbour codes to the contour length or surface
        area of the corresponding contour and as its second element the kernel used to compute the neighbour codes.

    """
    if isinstance(spacing, tuple) and len(spacing) == 2:
        return table_contour_length(spacing, device)
    if isinstance(spacing, tuple) and len(spacing) == 3:
        return table_surface_area(spacing, device)
    raise ValueError('The spacing must be a tuple of length 2 or 3.')