from typing import Optional, Tuple, Union
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.utilities import rank_zero_warn, reduce
Compute the peak signal-to-noise ratio.

    Args:
        preds: estimated signal
        target: groun truth signal
        data_range:
            the range of the data. If None, it is determined from the data (max - min). If a tuple is provided then
            the range is calculated as the difference and input is clamped between the values.
            The ``data_range`` must be given when ``dim`` is not None.
        base: a base of a logarithm to use
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'`` or None``: no reduction will be applied

        dim:
            Dimensions to reduce PSNR scores over provided as either an integer or a list of integers. Default is
            None meaning scores will be reduced across all dimensions.

    Return:
        Tensor with PSNR score

    Raises:
        ValueError:
            If ``dim`` is not ``None`` and ``data_range`` is not provided.

    Example:
        >>> from torchmetrics.functional.image import peak_signal_noise_ratio
        >>> pred = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        >>> target = torch.tensor([[3.0, 2.0], [1.0, 0.0]])
        >>> peak_signal_noise_ratio(pred, target)
        tensor(2.5527)

    .. note::
        Half precision is only support on GPU for this metric

    