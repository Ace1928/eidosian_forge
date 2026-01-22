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
def handleSenderKeyMessage(self, node):
    encMessageProtocolEntity = EncryptedMessageProtocolEntity.fromProtocolTreeNode(node)
    enc = encMessageProtocolEntity.getEnc(EncProtocolEntity.TYPE_SKMSG)
    try:
        plaintext = self.manager.group_decrypt(groupid=encMessageProtocolEntity.getFrom(True), participantid=encMessageProtocolEntity.getParticipant(False), data=enc.getData())
        self.parseAndHandleMessageProto(encMessageProtocolEntity, plaintext)
        node = encMessageProtocolEntity.toProtocolTreeNode()
        node.addChild(ProtoProtocolEntity(plaintext, enc.getMediaType()).toProtocolTreeNode())
        self.toUpper(node)
    except exceptions.NoSessionException:
        logger.warning('No session for %s, going to send a retry', encMessageProtocolEntity.getAuthor(False))
        self.send_retry(node, self.manager.registration_id)