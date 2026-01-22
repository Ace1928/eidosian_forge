from yowsup.layers.network.dispatcher.dispatcher import YowConnectionDispatcher
import asyncore
import logging
import socket
import traceback
def handle_close(self):
    logger.debug('handle_close')
    self.close()
    self._connected = False
    self.connectionCallbacks.onDisconnected()