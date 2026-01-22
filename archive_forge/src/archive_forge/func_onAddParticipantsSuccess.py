from yowsup.layers import YowLayer, YowLayerEvent, YowProtocolLayer
from yowsup.layers.protocol_iq.protocolentities import ErrorIqProtocolEntity
from yowsup.layers.protocol_iq.protocolentities.iq_result import ResultIqProtocolEntity
from .protocolentities import *
import logging
def onAddParticipantsSuccess(self, node, originalIqEntity):
    logger.info('Group add participants success')
    self.toUpper(SuccessAddParticipantsIqProtocolEntity.fromProtocolTreeNode(node))