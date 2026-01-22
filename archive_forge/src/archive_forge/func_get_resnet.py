import os
from ....context import cpu
from ...block import HybridBlock
from ... import nn
from .... import base
from .... util import is_np_array
def get_resnet(version, num_layers, pretrained=False, ctx=cpu(), root=os.path.join(base.data_dir(), 'models'), **kwargs):
    """ResNet V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    ResNet V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    version : int
        Version of ResNet. Options are 1, 2.
    num_layers : int
        Numbers of layers. Options are 18, 34, 50, 101, 152.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    """
    assert num_layers in resnet_spec, 'Invalid number of layers: %d. Options are %s' % (num_layers, str(resnet_spec.keys()))
    block_type, layers, channels = resnet_spec[num_layers]
    assert version >= 1 and version <= 2, 'Invalid resnet version: %d. Options are 1 and 2.' % version
    resnet_class = resnet_net_versions[version - 1]
    block_class = resnet_block_versions[version - 1][block_type]
    net = resnet_class(block_class, layers, channels, **kwargs)
    if pretrained:
        from ..model_store import get_model_file
        net.load_parameters(get_model_file('resnet%d_v%d' % (num_layers, version), root=root), ctx=ctx)
    return net