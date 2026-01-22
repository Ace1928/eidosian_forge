from typing import Tuple
import torch
from torch import Tensor
from torchmetrics.functional.regression.utils import _check_data_shape_to_num_outputs
from torchmetrics.utilities.checks import _check_same_shape
def log_cosh_error(preds: Tensor, target: Tensor) -> Tensor:
    """Compute the `LogCosh Error`_.

    .. math:: \\text{LogCoshError} = \\log\\left(\\frac{\\exp(\\hat{y} - y) + \\exp(\\hat{y - y})}{2}\\right)

    Where :math:`y` is a tensor of target values, and :math:`\\hat{y}` is a tensor of predictions.

    Args:
        preds: estimated labels with shape ``(batch_size,)`` or `(batch_size, num_outputs)``
        target: ground truth labels with shape ``(batch_size,)`` or `(batch_size, num_outputs)``

    Return:
        Tensor with LogCosh error

    Example (single output regression)::
        >>> from torchmetrics.functional.regression import log_cosh_error
        >>> preds = torch.tensor([3.0, 5.0, 2.5, 7.0])
        >>> target = torch.tensor([2.5, 5.0, 4.0, 8.0])
        >>> log_cosh_error(preds, target)
        tensor(0.3523)

    Example (multi output regression)::
        >>> from torchmetrics.functional.regression import log_cosh_error
        >>> preds = torch.tensor([[3.0, 5.0, 1.2], [-2.1, 2.5, 7.0]])
        >>> target = torch.tensor([[2.5, 5.0, 1.3], [0.3, 4.0, 8.0]])
        >>> log_cosh_error(preds, target)
        tensor([0.9176, 0.4277, 0.2194])

    """
    sum_log_cosh_error, num_obs = _log_cosh_error_update(preds, target, num_outputs=1 if preds.ndim == 1 else preds.shape[-1])
    return _log_cosh_error_compute(sum_log_cosh_error, num_obs)