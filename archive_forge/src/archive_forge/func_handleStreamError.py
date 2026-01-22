from yowsup.common import YowConstants
from yowsup.layers import YowLayerEvent, YowProtocolLayer, EventCallback
from yowsup.layers.network import YowNetworkLayer
from .protocolentities import *
from .layer_interface_authentication import YowAuthenticationProtocolLayerInterface
from .protocolentities import StreamErrorProtocolEntity
import logging
def handleStreamError(self, node):
    nodeEntity = StreamErrorProtocolEntity.fromProtocolTreeNode(node)
    errorType = nodeEntity.getErrorType()
    if not errorType:
        raise NotImplementedError('Unhandled stream:error node:\n%s' % node)
    self.toUpper(nodeEntity)