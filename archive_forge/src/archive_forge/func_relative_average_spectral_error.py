from typing import Tuple
import torch
from torch import Tensor
from torchmetrics.functional.image.rmse_sw import _rmse_sw_compute, _rmse_sw_update
from torchmetrics.functional.image.utils import _uniform_filter
def relative_average_spectral_error(preds: Tensor, target: Tensor, window_size: int=8) -> Tensor:
    """Compute Relative Average Spectral Error (RASE) (RelativeAverageSpectralError_).

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

    """
    if not isinstance(window_size, int) or (isinstance(window_size, int) and window_size < 1):
        raise ValueError('Argument `window_size` is expected to be a positive integer.')
    img_shape = target.shape[1:]
    rmse_map = torch.zeros(img_shape, dtype=target.dtype, device=target.device)
    target_sum = torch.zeros(img_shape, dtype=target.dtype, device=target.device)
    total_images = torch.tensor(0.0, device=target.device)
    rmse_map, target_sum, total_images = _rase_update(preds, target, window_size, rmse_map, target_sum, total_images)
    return _rase_compute(rmse_map, target_sum, total_images, window_size)