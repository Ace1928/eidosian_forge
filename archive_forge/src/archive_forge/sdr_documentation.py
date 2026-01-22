import math
from typing import Optional, Tuple
import torch
from torch import Tensor
from torch.linalg import norm
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.imports import _FAST_BSS_EVAL_AVAILABLE
`Source-aggregated signal-to-distortion ratio`_ (SA-SDR).

    The SA-SDR is proposed to provide a stable gradient for meeting style source separation, where
    one-speaker and multiple-speaker scenes coexist.

    Args:
        preds: float tensor with shape ``(..., spk, time)``
        target: float tensor with shape ``(..., spk, time)``
        scale_invariant: if True, scale the targets of different speakers with the same alpha
        zero_mean: If to zero mean target and preds or not

    Returns:
        SA-SDR with shape ``(...)``

    Example:
        >>> import torch
        >>> from torchmetrics.functional.audio import source_aggregated_signal_distortion_ratio
        >>> g = torch.manual_seed(1)
        >>> preds = torch.randn(2, 8000)  # [..., spk, time]
        >>> target = torch.randn(2, 8000)
        >>> source_aggregated_signal_distortion_ratio(preds, target)
        tensor(-41.6579)
        >>> # use with permutation_invariant_training
        >>> from torchmetrics.functional.audio import permutation_invariant_training
        >>> preds = torch.randn(4, 2, 8000)  # [batch, spk, time]
        >>> target = torch.randn(4, 2, 8000)
        >>> best_metric, best_perm = permutation_invariant_training(preds, target,
        ...     source_aggregated_signal_distortion_ratio, mode="permutation-wise")
        >>> best_metric
        tensor([-37.9511, -41.9124, -42.7369, -42.5155])
        >>> best_perm
        tensor([[1, 0],
                [1, 0],
                [0, 1],
                [1, 0]])

    