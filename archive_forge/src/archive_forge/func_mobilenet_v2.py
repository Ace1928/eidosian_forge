from functools import partial
from typing import Any, Optional, Union
from torch import nn, Tensor
from torch.ao.quantization import DeQuantStub, QuantStub
from torchvision.models.mobilenetv2 import InvertedResidual, MobileNet_V2_Weights, MobileNetV2
from ...ops.misc import Conv2dNormActivation
from ...transforms._presets import ImageClassification
from .._api import register_model, Weights, WeightsEnum
from .._meta import _IMAGENET_CATEGORIES
from .._utils import _ovewrite_named_param, handle_legacy_interface
from .utils import _fuse_modules, _replace_relu, quantize_model
@register_model(name='quantized_mobilenet_v2')
@handle_legacy_interface(weights=('pretrained', lambda kwargs: MobileNet_V2_QuantizedWeights.IMAGENET1K_QNNPACK_V1 if kwargs.get('quantize', False) else MobileNet_V2_Weights.IMAGENET1K_V1))
def mobilenet_v2(*, weights: Optional[Union[MobileNet_V2_QuantizedWeights, MobileNet_V2_Weights]]=None, progress: bool=True, quantize: bool=False, **kwargs: Any) -> QuantizableMobileNetV2:
    """
    Constructs a MobileNetV2 architecture from
    `MobileNetV2: Inverted Residuals and Linear Bottlenecks
    <https://arxiv.org/abs/1801.04381>`_.

    .. note::
        Note that ``quantize = True`` returns a quantized model with 8 bit
        weights. Quantized models only support inference and run on CPUs.
        GPU inference is not yet supported.

    Args:
        weights (:class:`~torchvision.models.quantization.MobileNet_V2_QuantizedWeights` or :class:`~torchvision.models.MobileNet_V2_Weights`, optional): The
            pretrained weights for the model. See
            :class:`~torchvision.models.quantization.MobileNet_V2_QuantizedWeights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        quantize (bool, optional): If True, returns a quantized version of the model. Default is False.
        **kwargs: parameters passed to the ``torchvision.models.quantization.QuantizableMobileNetV2``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/quantization/mobilenetv2.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.quantization.MobileNet_V2_QuantizedWeights
        :members:
    .. autoclass:: torchvision.models.MobileNet_V2_Weights
        :members:
        :noindex:
    """
    weights = (MobileNet_V2_QuantizedWeights if quantize else MobileNet_V2_Weights).verify(weights)
    if weights is not None:
        _ovewrite_named_param(kwargs, 'num_classes', len(weights.meta['categories']))
        if 'backend' in weights.meta:
            _ovewrite_named_param(kwargs, 'backend', weights.meta['backend'])
    backend = kwargs.pop('backend', 'qnnpack')
    model = QuantizableMobileNetV2(block=QuantizableInvertedResidual, **kwargs)
    _replace_relu(model)
    if quantize:
        quantize_model(model, backend)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
    return model