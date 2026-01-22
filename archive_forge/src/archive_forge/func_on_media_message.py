from yowsup.layers.interface import YowInterfaceLayer, ProtocolEntityCallback
from yowsup.layers.protocol_media.protocolentities import *
from yowsup.layers.network.layer import YowNetworkLayer
from yowsup.layers import EventCallback
import sys
from ..common.sink_worker import SinkWorker
import tempfile
import logging
import os
def on_media_message(self, media_message_protocolentity):
    self._sink_worker.enqueue(media_message_protocolentity)