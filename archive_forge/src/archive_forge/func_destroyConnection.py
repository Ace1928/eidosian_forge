from yowsup.layers import YowLayer, YowLayerEvent, EventCallback
from yowsup.layers.network.layer_interface import YowNetworkLayerInterface
from yowsup.layers.network.dispatcher.dispatcher import ConnectionCallbacks
from yowsup.layers.network.dispatcher.dispatcher import YowConnectionDispatcher
from yowsup.layers.network.dispatcher.dispatcher_socket import SocketConnectionDispatcher
from yowsup.layers.network.dispatcher.dispatcher_asyncore import AsyncoreConnectionDispatcher
import logging
def destroyConnection(self, reason=None):
    self._disconnect_reason = reason
    self.state = self.__class__.STATE_DISCONNECTING
    self._dispatcher.disconnect()