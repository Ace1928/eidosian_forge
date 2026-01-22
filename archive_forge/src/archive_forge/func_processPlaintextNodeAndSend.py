from yowsup.layers.protocol_messages.proto.e2e_pb2 import Message
from yowsup.layers.axolotl.protocolentities import *
from yowsup.layers.auth.layer_authentication import YowAuthenticationProtocolLayer
from yowsup.layers.protocol_groups.protocolentities import InfoGroupsIqProtocolEntity, InfoGroupsResultIqProtocolEntity
from axolotl.protocol.whispermessage import WhisperMessage
from yowsup.layers.protocol_messages.protocolentities.message import MessageMetaAttributes
from yowsup.layers.axolotl.protocolentities.iq_keys_get_result import MissingParametersException
from yowsup.axolotl import exceptions
from .layer_base import AxolotlBaseLayer
import logging
def processPlaintextNodeAndSend(self, node, retryReceiptEntity=None):
    recipient_id = node['to'].split('@')[0]
    isGroup = '-' in recipient_id

    def on_get_keys_error(error_node, getkeys_entity, plaintext_node):
        logger.error('Failed to fetch keys for %s, is that a valid user? Server response: [code=%s, text=%s], aborting send.' % (plaintext_node['to'], error_node.children[0]['code'], error_node.children[0]['text']))

    def on_get_keys_success(node, success_jids, errors):
        if len(errors):
            self.on_get_keys_process_errors(errors)
        elif len(success_jids) == 1:
            self.sendToContact(node)
        else:
            raise NotImplementedError()
    if isGroup:
        self.sendToGroup(node, retryReceiptEntity)
    elif self.manager.session_exists(recipient_id):
        self.sendToContact(node)
    else:
        self.getKeysFor([node['to']], lambda successJids, errors: on_get_keys_success(node, successJids, errors), lambda error_node, entity: on_get_keys_error(error_node, entity, node))