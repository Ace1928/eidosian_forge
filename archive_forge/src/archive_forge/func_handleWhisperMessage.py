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
def handleWhisperMessage(self, node):
    encMessageProtocolEntity = EncryptedMessageProtocolEntity.fromProtocolTreeNode(node)
    enc = encMessageProtocolEntity.getEnc(EncProtocolEntity.TYPE_MSG)
    plaintext = self.manager.decrypt_msg(encMessageProtocolEntity.getAuthor(False), enc.getData(), enc.getVersion() == 2)
    if enc.getVersion() == 2:
        self.parseAndHandleMessageProto(encMessageProtocolEntity, plaintext)
    node = encMessageProtocolEntity.toProtocolTreeNode()
    node.addChild(ProtoProtocolEntity(plaintext, enc.getMediaType()).toProtocolTreeNode())
    self.toUpper(node)