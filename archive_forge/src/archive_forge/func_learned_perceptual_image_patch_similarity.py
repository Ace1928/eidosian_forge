import inspect
import os
from typing import List, NamedTuple, Optional, Tuple, Union
import torch
from torch import Tensor, nn
from typing_extensions import Literal
from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE, _TORCHVISION_GREATER_EQUAL_0_13
def learned_perceptual_image_patch_similarity(img1: Tensor, img2: Tensor, net_type: Literal['alex', 'vgg', 'squeeze']='alex', reduction: Literal['sum', 'mean']='mean', normalize: bool=False) -> Tensor:
    """The Learned Perceptual Image Patch Similarity (`LPIPS_`) calculates perceptual similarity between two images.

    LPIPS essentially computes the similarity between the activations of two image patches for some pre-defined network.
    This measure has been shown to match human perception well. A low LPIPS score means that image patches are
    perceptual similar.

    Both input image patches are expected to have shape ``(N, 3, H, W)``. The minimum size of `H, W` depends on the
    chosen backbone (see `net_type` arg).

    Args:
        img1: first set of images
        img2: second set of images
        net_type: str indicating backbone network type to use. Choose between `'alex'`, `'vgg'` or `'squeeze'`
        reduction: str indicating how to reduce over the batch dimension. Choose between `'sum'` or `'mean'`.
        normalize: by default this is ``False`` meaning that the input is expected to be in the [-1,1] range. If set
            to ``True`` will instead expect input to be in the ``[0,1]`` range.

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(123)
        >>> from torchmetrics.functional.image.lpips import learned_perceptual_image_patch_similarity
        >>> img1 = (torch.rand(10, 3, 100, 100) * 2) - 1
        >>> img2 = (torch.rand(10, 3, 100, 100) * 2) - 1
        >>> learned_perceptual_image_patch_similarity(img1, img2, net_type='squeeze')
        tensor(0.1008)

    """
    net = _NoTrainLpips(net=net_type).to(device=img1.device, dtype=img1.dtype)
    loss, total = _lpips_update(img1, img2, net, normalize)
    return _lpips_compute(loss.sum(), total, reduction)