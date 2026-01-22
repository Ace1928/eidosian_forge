import math
from typing import Optional, Tuple
import torch
from torch import Tensor
from torch.linalg import norm
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.imports import _FAST_BSS_EVAL_AVAILABLE
def signal_distortion_ratio(preds: Tensor, target: Tensor, use_cg_iter: Optional[int]=None, filter_length: int=512, zero_mean: bool=False, load_diag: Optional[float]=None) -> Tensor:
    """Calculate Signal to Distortion Ratio (SDR) metric. See `SDR ref1`_ and `SDR ref2`_ for details on the metric.

    .. note:
        The metric currently does not seem to work with Pytorch v1.11 and specific GPU hardware.

    Args:
        preds: float tensor with shape ``(...,time)``
        target: float tensor with shape ``(...,time)``
        use_cg_iter:
            If provided, conjugate gradient descent is used to solve for the distortion
            filter coefficients instead of direct Gaussian elimination, which requires that
            ``fast-bss-eval`` is installed and pytorch version >= 1.8.
            This can speed up the computation of the metrics in case the filters
            are long. Using a value of 10 here has been shown to provide
            good accuracy in most cases and is sufficient when using this
            loss to train neural separation networks.
        filter_length: The length of the distortion filter allowed
        zero_mean: When set to True, the mean of all signals is subtracted prior to computation of the metrics
        load_diag:
            If provided, this small value is added to the diagonal coefficients of
            the system metrics when solving for the filter coefficients.
            This can help stabilize the metric in the case where some reference signals may sometimes be zero

    Returns:
        Float tensor with shape ``(...,)`` of SDR values per sample

    Raises:
        RuntimeError:
            If ``preds`` and ``target`` does not have the same shape

    Example:
        >>> import torch
        >>> from torchmetrics.functional.audio import signal_distortion_ratio
        >>> g = torch.manual_seed(1)
        >>> preds = torch.randn(8000)
        >>> target = torch.randn(8000)
        >>> signal_distortion_ratio(preds, target)
        tensor(-12.0589)
        >>> # use with permutation_invariant_training
        >>> from torchmetrics.functional.audio import permutation_invariant_training
        >>> preds = torch.randn(4, 2, 8000)  # [batch, spk, time]
        >>> target = torch.randn(4, 2, 8000)
        >>> best_metric, best_perm = permutation_invariant_training(preds, target, signal_distortion_ratio)
        >>> best_metric
        tensor([-11.6375, -11.4358, -11.7148, -11.6325])
        >>> best_perm
        tensor([[1, 0],
                [0, 1],
                [1, 0],
                [0, 1]])

    """
    _check_same_shape(preds, target)
    preds_dtype = preds.dtype
    preds = preds.double()
    target = target.double()
    if zero_mean:
        preds = preds - preds.mean(dim=-1, keepdim=True)
        target = target - target.mean(dim=-1, keepdim=True)
    target = target / torch.clamp(norm(target, dim=-1, keepdim=True), min=1e-06)
    preds = preds / torch.clamp(norm(preds, dim=-1, keepdim=True), min=1e-06)
    r_0, b = _compute_autocorr_crosscorr(target, preds, corr_len=filter_length)
    if load_diag is not None:
        r_0[..., 0] += load_diag
    if use_cg_iter is not None and _FAST_BSS_EVAL_AVAILABLE:
        from fast_bss_eval.torch.cgd import toeplitz_conjugate_gradient
        sol = toeplitz_conjugate_gradient(r_0, b, n_iter=use_cg_iter)
    else:
        if use_cg_iter is not None and (not _FAST_BSS_EVAL_AVAILABLE):
            rank_zero_warn('The `use_cg_iter` parameter of `SDR` requires that `fast-bss-eval` is installed. To make this this warning disappear, you could install `fast-bss-eval` using `pip install fast-bss-eval` or set `use_cg_iter=None`. For this time, the solver provided by Pytorch is used.', UserWarning)
        r = _symmetric_toeplitz(r_0)
        sol = torch.linalg.solve(r, b)
    coh = torch.einsum('...l,...l->...', b, sol)
    ratio = coh / (1 - coh)
    val = 10.0 * torch.log10(ratio)
    if preds_dtype == torch.float64:
        return val
    return val.float()