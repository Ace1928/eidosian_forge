from .layer_base import AxolotlBaseLayer
from yowsup.layers.protocol_receipts.protocolentities import OutgoingReceiptProtocolEntity
from yowsup.layers.protocol_messages.proto.e2e_pb2 import *
from yowsup.layers.axolotl.protocolentities import *
from yowsup.structs import ProtocolTreeNode
from yowsup.layers.protocol_messages.protocolentities.proto import ProtoProtocolEntity
from yowsup.layers.axolotl.props import PROP_IDENTITY_AUTOTRUST
from yowsup.axolotl import exceptions
from axolotl.untrustedidentityexception import UntrustedIdentityException
import logging
def parseAndHandleMessageProto(self, encMessageProtocolEntity, serializedData):
    m = Message()
    try:
        m.ParseFromString(serializedData)
    except:
        print('DUMP:')
        print(serializedData)
        print([s for s in serializedData])
        raise
    if not m or not serializedData:
        raise exceptions.InvalidMessageException()
    if m.HasField('sender_key_distribution_message'):
        self.handleSenderKeyDistributionMessage(m.sender_key_distribution_message, encMessageProtocolEntity.getParticipant(False))