import torch
from torch import Tensor
from torchmetrics.functional.audio.sdr import scale_invariant_signal_distortion_ratio
from torchmetrics.utilities.checks import _check_same_shape
def signal_noise_ratio(preds: Tensor, target: Tensor, zero_mean: bool=False) -> Tensor:
    """Calculate `Signal-to-noise ratio`_ (SNR_) meric for evaluating quality of audio.

    .. math::
        \\text{SNR} = \\frac{P_{signal}}{P_{noise}}

    where  :math:`P` denotes the power of each signal. The SNR metric compares the level of the desired signal to
    the level of background noise. Therefore, a high value of SNR means that the audio is clear.

    Args:
        preds: float tensor with shape ``(...,time)``
        target: float tensor with shape ``(...,time)``
        zero_mean: if to zero mean target and preds or not

    Returns:
        Float tensor with shape ``(...,)`` of SNR values per sample

    Raises:
        RuntimeError:
            If ``preds`` and ``target`` does not have the same shape

    Example:
        >>> from torchmetrics.functional.audio import signal_noise_ratio
        >>> target = torch.tensor([3.0, -0.5, 2.0, 7.0])
        >>> preds = torch.tensor([2.5, 0.0, 2.0, 8.0])
        >>> signal_noise_ratio(preds, target)
        tensor(16.1805)

    """
    _check_same_shape(preds, target)
    eps = torch.finfo(preds.dtype).eps
    if zero_mean:
        target = target - torch.mean(target, dim=-1, keepdim=True)
        preds = preds - torch.mean(preds, dim=-1, keepdim=True)
    noise = target - preds
    snr_value = (torch.sum(target ** 2, dim=-1) + eps) / (torch.sum(noise ** 2, dim=-1) + eps)
    return 10 * torch.log10(snr_value)