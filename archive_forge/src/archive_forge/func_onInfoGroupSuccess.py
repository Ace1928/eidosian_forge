from yowsup.layers import YowLayer, YowLayerEvent, YowProtocolLayer
from yowsup.layers.protocol_iq.protocolentities import ErrorIqProtocolEntity
from yowsup.layers.protocol_iq.protocolentities.iq_result import ResultIqProtocolEntity
from .protocolentities import *
import logging
def onInfoGroupSuccess(self, node, originalIqEntity):
    logger.info('Group info success')
    self.toUpper(InfoGroupsResultIqProtocolEntity.fromProtocolTreeNode(node))