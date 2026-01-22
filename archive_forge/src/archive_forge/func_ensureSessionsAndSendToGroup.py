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
def ensureSessionsAndSendToGroup(self, node, jids):
    logger.debug('ensureSessionsAndSendToGroup(node=[omitted], jids=%s)' % jids)
    jidsNoSession = []
    for jid in jids:
        if not self.manager.session_exists(jid.split('@')[0]):
            jidsNoSession.append(jid)

    def on_get_keys_success(node, success_jids, errors):
        if len(errors):
            self.on_get_keys_process_errors(errors)
        self.sendToGroupWithSessions(node, success_jids)
    if len(jidsNoSession):
        self.getKeysFor(jidsNoSession, lambda successJids, errors: on_get_keys_success(node, successJids, errors))
    else:
        self.sendToGroupWithSessions(node, jids)