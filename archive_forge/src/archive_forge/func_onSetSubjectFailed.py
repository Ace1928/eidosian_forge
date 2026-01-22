from yowsup.layers import YowLayer, YowLayerEvent, YowProtocolLayer
from yowsup.layers.protocol_iq.protocolentities import ErrorIqProtocolEntity
from yowsup.layers.protocol_iq.protocolentities.iq_result import ResultIqProtocolEntity
from .protocolentities import *
import logging
def onSetSubjectFailed(self, node, originalIqEntity):
    logger.error('Group subject change failed')
    self.toUpper(ErrorIqProtocolEntity.fromProtocolTreeNode(node))