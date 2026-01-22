from yowsup.layers import YowProtocolLayer
from .protocolentities import * 
def recvAckNode(self, node):
    self.toUpper(IncomingAckProtocolEntity.fromProtocolTreeNode(node))