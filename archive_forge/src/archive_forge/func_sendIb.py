from yowsup.layers import YowLayer, YowLayerEvent, YowProtocolLayer
from .protocolentities import *
import logging
def sendIb(self, entity):
    if entity.__class__ == CleanIqProtocolEntity:
        self.toLower(entity.toProtocolTreeNode())