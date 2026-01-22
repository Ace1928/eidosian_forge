from yowsup.layers import YowLayer, YowLayerEvent, YowProtocolLayer
from yowsup.layers.protocol_iq.protocolentities import ErrorIqProtocolEntity
from yowsup.layers.protocol_iq.protocolentities.iq_result import ResultIqProtocolEntity
from .protocolentities import *
import logging
def onSetSubjectSuccess(self, node, originalIqEntity):
    logger.info('Group subject change success')
    self.toUpper(ResultIqProtocolEntity.fromProtocolTreeNode(node))