from yowsup.layers import YowLayer, YowLayerEvent, YowProtocolLayer
from .protocolentities import *
from yowsup.layers.protocol_acks.protocolentities import OutgoingAckProtocolEntity
import logging
def sendNotification(self, entity):
    if entity.getTag() == 'notification':
        self.toLower(entity.toProtocolTreeNode())