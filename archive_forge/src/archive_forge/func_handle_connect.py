from yowsup.layers.network.dispatcher.dispatcher import YowConnectionDispatcher
import asyncore
import logging
import socket
import traceback
def handle_connect(self):
    logger.debug('handle_connect')
    if not self._connected:
        self._connected = True
        self.connectionCallbacks.onConnected()