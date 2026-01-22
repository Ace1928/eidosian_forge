from typing import Optional, Tuple, Union
from torch import Tensor
from typing_extensions import Literal
Compute total variation loss.

    Args:
        img: A `Tensor` of shape `(N, C, H, W)` consisting of images
        reduction: a method to reduce metric score over samples.

            - ``'mean'``: takes the mean over samples
            - ``'sum'``: takes the sum over samples
            - ``None`` or ``'none'``: return the score per sample

    Returns:
        A loss scalar value containing the total variation

    Raises:
        ValueError:
            If ``reduction`` is not one of ``'sum'``, ``'mean'``, ``'none'`` or ``None``
        RuntimeError:
            If ``img`` is not 4D tensor

    Example:
        >>> import torch
        >>> from torchmetrics.functional.image import total_variation
        >>> _ = torch.manual_seed(42)
        >>> img = torch.rand(5, 3, 28, 28)
        >>> total_variation(img)
        tensor(7546.8018)

    