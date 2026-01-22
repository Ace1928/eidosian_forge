from functools import partial
from typing import Any, List, Optional, Type, Union
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.resnet import (
from ...transforms._presets import ImageClassification
from .._api import register_model, Weights, WeightsEnum
from .._meta import _IMAGENET_CATEGORIES
from .._utils import _ovewrite_named_param, handle_legacy_interface
from .utils import _fuse_modules, _replace_relu, quantize_model
@register_model(name='quantized_resnet18')
@handle_legacy_interface(weights=('pretrained', lambda kwargs: ResNet18_QuantizedWeights.IMAGENET1K_FBGEMM_V1 if kwargs.get('quantize', False) else ResNet18_Weights.IMAGENET1K_V1))
def resnet18(*, weights: Optional[Union[ResNet18_QuantizedWeights, ResNet18_Weights]]=None, progress: bool=True, quantize: bool=False, **kwargs: Any) -> QuantizableResNet:
    """ResNet-18 model from
    `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`_

    .. note::
        Note that ``quantize = True`` returns a quantized model with 8 bit
        weights. Quantized models only support inference and run on CPUs.
        GPU inference is not yet supported.

    Args:
        weights (:class:`~torchvision.models.quantization.ResNet18_QuantizedWeights` or :class:`~torchvision.models.ResNet18_Weights`, optional): The
            pretrained weights for the model. See
            :class:`~torchvision.models.quantization.ResNet18_QuantizedWeights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        quantize (bool, optional): If True, return a quantized version of the model. Default is False.
        **kwargs: parameters passed to the ``torchvision.models.quantization.QuantizableResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/quantization/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.quantization.ResNet18_QuantizedWeights
        :members:

    .. autoclass:: torchvision.models.ResNet18_Weights
        :members:
        :noindex:
    """
    weights = (ResNet18_QuantizedWeights if quantize else ResNet18_Weights).verify(weights)
    return _resnet(QuantizableBasicBlock, [2, 2, 2, 2], weights, progress, quantize, **kwargs)