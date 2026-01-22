from yowsup.layers import YowProtocolLayer
from .protocolentities import *
from yowsup.layers.protocol_acks.protocolentities import OutgoingAckProtocolEntity
from yowsup.layers.protocol_receipts.protocolentities import OutgoingReceiptProtocolEntity
def recvCall(self, node):
    entity = CallProtocolEntity.fromProtocolTreeNode(node)
    if entity.getType() == 'offer':
        receipt = OutgoingReceiptProtocolEntity(node['id'], node['from'], callId=entity.getCallId())
        self.toLower(receipt.toProtocolTreeNode())
    else:
        ack = OutgoingAckProtocolEntity(node['id'], 'call', None, node['from'])
        self.toLower(ack.toProtocolTreeNode())
    self.toUpper(entity)