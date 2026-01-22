from yowsup.layers.interface import YowInterfaceLayer, ProtocolEntityCallback
from yowsup.layers.protocol_media.protocolentities import *
from yowsup.layers.network.layer import YowNetworkLayer
from yowsup.layers import EventCallback
import sys
from ..common.sink_worker import SinkWorker
import tempfile
import logging
import os
@ProtocolEntityCallback('message')
def on_message(self, message_protocolentity):
    self.toLower(message_protocolentity.ack())
    self.toLower(message_protocolentity.ack(True))
    if isinstance(message_protocolentity, MediaMessageProtocolEntity):
        self.on_media_message(message_protocolentity)