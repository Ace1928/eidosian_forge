from typing import Optional, Union
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
def calculate_generalized_mean(x: Tensor, p: Union[int, Literal['min', 'geometric', 'arithmetic', 'max']]) -> Tensor:
    """Return generalized (power) mean of a tensor.

    Args:
        x: tensor
        p: power

    Returns:
        generalized_mean: generalized mean

    Example (p="min"):
        >>> from torchmetrics.functional.clustering.utils import calculate_generalized_mean
        >>> x = torch.tensor([1, 3, 2, 2, 1])
        >>> calculate_generalized_mean(x, "min")
        tensor(1)

    Example (p="geometric"):
        >>> from torchmetrics.functional.clustering.utils import calculate_generalized_mean
        >>> x = torch.tensor([1, 3, 2, 2, 1])
        >>> calculate_generalized_mean(x, "geometric")
        tensor(1.6438)

    """
    if torch.is_complex(x) or not is_nonnegative(x):
        raise ValueError('`x` must contain positive real numbers')
    if isinstance(p, str):
        if p == 'min':
            return x.min()
        if p == 'geometric':
            return torch.exp(torch.mean(x.log()))
        if p == 'arithmetic':
            return x.mean()
        if p == 'max':
            return x.max()
        raise ValueError("'method' must be 'min', 'geometric', 'arirthmetic', or 'max'")
    return torch.mean(torch.pow(x, p)) ** (1.0 / p)