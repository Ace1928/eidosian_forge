import os
from ... import nn
from ....context import cpu
from ...block import HybridBlock
from .... import base
def mobilenet1_0(**kwargs):
    """MobileNet model from the
    `"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
    <https://arxiv.org/abs/1704.04861>`_ paper, with width multiplier 1.0.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    """
    return get_mobilenet(1.0, **kwargs)