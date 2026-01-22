from typing import Tuple
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.image.uqi import universal_image_quality_index
from torchmetrics.utilities.distributed import reduce
def spectral_distortion_index(preds: Tensor, target: Tensor, p: int=1, reduction: Literal['elementwise_mean', 'sum', 'none']='elementwise_mean') -> Tensor:
    """Calculate `Spectral Distortion Index`_ (SpectralDistortionIndex_) also known as D_lambda.

    Metric is used to compare the spectral distortion between two images.

    Args:
        preds: Low resolution multispectral image
        target: High resolution fused image
        p: Large spectral differences
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

    Return:
        Tensor with SpectralDistortionIndex score

    Raises:
        TypeError:
            If ``preds`` and ``target`` don't have the same data type.
        ValueError:
            If ``preds`` and ``target`` don't have ``BxCxHxW shape``.
        ValueError:
            If ``p`` is not a positive integer.

    Example:
        >>> from torchmetrics.functional.image import spectral_distortion_index
        >>> _ = torch.manual_seed(42)
        >>> preds = torch.rand([16, 3, 16, 16])
        >>> target = torch.rand([16, 3, 16, 16])
        >>> spectral_distortion_index(preds, target)
        tensor(0.0234)

    """
    if not isinstance(p, int) or p <= 0:
        raise ValueError(f'Expected `p` to be a positive integer. Got p: {p}.')
    preds, target = _spectral_distortion_index_update(preds, target)
    return _spectral_distortion_index_compute(preds, target, p, reduction)