from yowsup.layers import YowLayer, YowLayerEvent, YowProtocolLayer
from .protocolentities import *
def sendIq(self, entity):
    if entity.getXmlns() == 'jabber:iq:privacy':
        self.entityToLower(entity)