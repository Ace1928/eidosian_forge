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
def handleEncMessage(self, node):
    encMessageProtocolEntity = EncryptedMessageProtocolEntity.fromProtocolTreeNode(node)
    isGroup = node['participant'] is not None
    senderJid = node['participant'] if isGroup else node['from']
    if node.getChild('enc')['v'] == '2' and node['from'] not in self.v2Jids:
        self.v2Jids.append(node['from'])
    try:
        if encMessageProtocolEntity.getEnc(EncProtocolEntity.TYPE_PKMSG):
            self.handlePreKeyWhisperMessage(node)
        elif encMessageProtocolEntity.getEnc(EncProtocolEntity.TYPE_MSG):
            self.handleWhisperMessage(node)
        if encMessageProtocolEntity.getEnc(EncProtocolEntity.TYPE_SKMSG):
            self.handleSenderKeyMessage(node)
        self.reset_retries(node['id'])
    except exceptions.InvalidKeyIdException:
        logger.warning('Invalid KeyId for %s, going to send a retry', encMessageProtocolEntity.getAuthor(False))
        self.send_retry(node, self.manager.registration_id)
    except exceptions.InvalidMessageException:
        logger.warning('InvalidMessage for %s, going to send a retry', encMessageProtocolEntity.getAuthor(False))
        self.send_retry(node, self.manager.registration_id)
    except exceptions.NoSessionException:
        logger.warning('No session for %s, getting their keys now', encMessageProtocolEntity.getAuthor(False))
        conversationIdentifier = (node['from'], node['participant'])
        if conversationIdentifier not in self.pendingIncomingMessages:
            self.pendingIncomingMessages[conversationIdentifier] = []
        self.pendingIncomingMessages[conversationIdentifier].append(node)
        successFn = lambda successJids, b: self.processPendingIncomingMessages(*conversationIdentifier) if len(successJids) else None
        self.getKeysFor([senderJid], successFn)
    except exceptions.DuplicateMessageException:
        logger.warning("Received a message that we've previously decrypted, going to send the delivery receipt myself")
        self.toLower(OutgoingReceiptProtocolEntity(node['id'], node['from'], participant=node['participant']).toProtocolTreeNode())
    except UntrustedIdentityException as e:
        if self.getProp(PROP_IDENTITY_AUTOTRUST, False):
            logger.warning('Autotrusting identity for %s', e.getName())
            self.manager.trust_identity(e.getName(), e.getIdentityKey())
            return self.handleEncMessage(node)
        else:
            logger.error('Ignoring message with untrusted identity')