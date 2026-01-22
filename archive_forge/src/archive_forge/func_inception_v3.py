import warnings
from functools import partial
from typing import Any, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import inception as inception_module
from torchvision.models.inception import Inception_V3_Weights, InceptionOutputs
from ...transforms._presets import ImageClassification
from .._api import register_model, Weights, WeightsEnum
from .._meta import _IMAGENET_CATEGORIES
from .._utils import _ovewrite_named_param, handle_legacy_interface
from .utils import _fuse_modules, _replace_relu, quantize_model
@register_model(name='quantized_inception_v3')
@handle_legacy_interface(weights=('pretrained', lambda kwargs: Inception_V3_QuantizedWeights.IMAGENET1K_FBGEMM_V1 if kwargs.get('quantize', False) else Inception_V3_Weights.IMAGENET1K_V1))
def inception_v3(*, weights: Optional[Union[Inception_V3_QuantizedWeights, Inception_V3_Weights]]=None, progress: bool=True, quantize: bool=False, **kwargs: Any) -> QuantizableInception3:
    """Inception v3 model architecture from
    `Rethinking the Inception Architecture for Computer Vision <http://arxiv.org/abs/1512.00567>`__.

    .. note::
        **Important**: In contrast to the other models the inception_v3 expects tensors with a size of
        N x 3 x 299 x 299, so ensure your images are sized accordingly.

    .. note::
        Note that ``quantize = True`` returns a quantized model with 8 bit
        weights. Quantized models only support inference and run on CPUs.
        GPU inference is not yet supported.

    Args:
        weights (:class:`~torchvision.models.quantization.Inception_V3_QuantizedWeights` or :class:`~torchvision.models.Inception_V3_Weights`, optional): The pretrained
            weights for the model. See
            :class:`~torchvision.models.quantization.Inception_V3_QuantizedWeights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr.
            Default is True.
        quantize (bool, optional): If True, return a quantized version of the model.
            Default is False.
        **kwargs: parameters passed to the ``torchvision.models.quantization.QuantizableInception3``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/quantization/inception.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.quantization.Inception_V3_QuantizedWeights
        :members:

    .. autoclass:: torchvision.models.Inception_V3_Weights
        :members:
        :noindex:
    """
    weights = (Inception_V3_QuantizedWeights if quantize else Inception_V3_Weights).verify(weights)
    original_aux_logits = kwargs.get('aux_logits', False)
    if weights is not None:
        if 'transform_input' not in kwargs:
            _ovewrite_named_param(kwargs, 'transform_input', True)
        _ovewrite_named_param(kwargs, 'aux_logits', True)
        _ovewrite_named_param(kwargs, 'num_classes', len(weights.meta['categories']))
        if 'backend' in weights.meta:
            _ovewrite_named_param(kwargs, 'backend', weights.meta['backend'])
    backend = kwargs.pop('backend', 'fbgemm')
    model = QuantizableInception3(**kwargs)
    _replace_relu(model)
    if quantize:
        quantize_model(model, backend)
    if weights is not None:
        if quantize and (not original_aux_logits):
            model.aux_logits = False
            model.AuxLogits = None
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
        if not quantize and (not original_aux_logits):
            model.aux_logits = False
            model.AuxLogits = None
    return model