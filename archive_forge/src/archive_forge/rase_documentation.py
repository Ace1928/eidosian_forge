from typing import Tuple
import torch
from torch import Tensor
from torchmetrics.functional.image.rmse_sw import _rmse_sw_compute, _rmse_sw_update
from torchmetrics.functional.image.utils import _uniform_filter
Compute Relative Average Spectral Error (RASE) (RelativeAverageSpectralError_).

    Args:
        preds: Deformed image
        target: Ground truth image
        window_size: Sliding window used for rmse calculation

    Return:
        Relative Average Spectral Error (RASE)

    Example:
        >>> from torchmetrics.functional.image import relative_average_spectral_error
        >>> g = torch.manual_seed(22)
        >>> preds = torch.rand(4, 3, 16, 16)
        >>> target = torch.rand(4, 3, 16, 16)
        >>> relative_average_spectral_error(preds, target)
        tensor(5114.66...)

    Raises:
        ValueError: If ``window_size`` is not a positive integer.

    