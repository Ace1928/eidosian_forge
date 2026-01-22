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
def processPendingIncomingMessages(self, jid, participantJid=None):
    conversationIdentifier = (jid, participantJid)
    if conversationIdentifier in self.pendingIncomingMessages:
        for messageNode in self.pendingIncomingMessages[conversationIdentifier]:
            self.onMessage(messageNode)
        del self.pendingIncomingMessages[conversationIdentifier]