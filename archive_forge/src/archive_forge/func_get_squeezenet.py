import os
from ....context import cpu
from ...block import HybridBlock
from ... import nn
from ...contrib.nn import HybridConcurrent
from .... import base
def get_squeezenet(version, pretrained=False, ctx=cpu(), root=os.path.join(base.data_dir(), 'models'), **kwargs):
    """SqueezeNet model from the `"SqueezeNet: AlexNet-level accuracy with 50x fewer parameters
    and <0.5MB model size" <https://arxiv.org/abs/1602.07360>`_ paper.
    SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.

    Parameters
    ----------
    version : str
        Version of squeezenet. Options are '1.0', '1.1'.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    """
    net = SqueezeNet(version, **kwargs)
    if pretrained:
        from ..model_store import get_model_file
        net.load_parameters(get_model_file('squeezenet%s' % version, root=root), ctx=ctx)
    return net