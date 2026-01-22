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
def sendToGroup(resultNode, requestEntity):
    groupInfo = InfoGroupsResultIqProtocolEntity.fromProtocolTreeNode(resultNode)
    jids = list(groupInfo.getParticipants().keys())
    if ownJid in jids:
        jids.remove(ownJid)
    return self.ensureSessionsAndSendToGroup(node, jids)