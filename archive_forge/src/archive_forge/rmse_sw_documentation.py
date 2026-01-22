from typing import Optional, Tuple, Union
import torch
from torch import Tensor
from torchmetrics.functional.image.utils import _uniform_filter
from torchmetrics.utilities.checks import _check_same_shape
Compute Root Mean Squared Error (RMSE) using sliding window.

    Args:
        preds: Deformed image
        target: Ground truth image
        window_size: Sliding window used for rmse calculation
        return_rmse_map: An indication whether the full rmse reduced image should be returned.

    Return:
        RMSE using sliding window
        (Optionally) RMSE map

    Example:
        >>> from torchmetrics.functional.image import root_mean_squared_error_using_sliding_window
        >>> g = torch.manual_seed(22)
        >>> preds = torch.rand(4, 3, 16, 16)
        >>> target = torch.rand(4, 3, 16, 16)
        >>> root_mean_squared_error_using_sliding_window(preds, target)
        tensor(0.3999)

    Raises:
        ValueError: If ``window_size`` is not a positive integer.

    