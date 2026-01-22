from yowsup.layers import YowProtocolLayer
from .protocolentities import * 
def sendAckEntity(self, entity):
    self.entityToLower(entity)