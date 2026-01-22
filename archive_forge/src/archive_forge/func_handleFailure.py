from yowsup.common import YowConstants
from yowsup.layers import YowLayerEvent, YowProtocolLayer, EventCallback
from yowsup.layers.network import YowNetworkLayer
from .protocolentities import *
from .layer_interface_authentication import YowAuthenticationProtocolLayerInterface
from .protocolentities import StreamErrorProtocolEntity
import logging
def handleFailure(self, node):
    nodeEntity = FailureProtocolEntity.fromProtocolTreeNode(node)
    self.toUpper(nodeEntity)
    self.broadcastEvent(YowLayerEvent(YowNetworkLayer.EVENT_STATE_DISCONNECT, reason='Authentication Failure'))