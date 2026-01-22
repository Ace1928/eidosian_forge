import os
from ....context import cpu
from ...block import HybridBlock
from ... import nn
from ...contrib.nn import HybridConcurrent, Identity
from .... import base
def get_densenet(num_layers, pretrained=False, ctx=cpu(), root=os.path.join(base.data_dir(), 'models'), **kwargs):
    """Densenet-BC model from the
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_ paper.

    Parameters
    ----------
    num_layers : int
        Number of layers for the variant of densenet. Options are 121, 161, 169, 201.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    """
    num_init_features, growth_rate, block_config = densenet_spec[num_layers]
    net = DenseNet(num_init_features, growth_rate, block_config, **kwargs)
    if pretrained:
        from ..model_store import get_model_file
        net.load_parameters(get_model_file('densenet%d' % num_layers, root=root), ctx=ctx)
    return net