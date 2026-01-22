from yowsup.layers import  YowProtocolLayer
from .protocolentities import *
from yowsup.layers.protocol_iq.protocolentities import ErrorIqProtocolEntity, ResultIqProtocolEntity
def onSetStatusResult(self, resultNode, originIqRequestEntity):
    self.toUpper(ResultIqProtocolEntity.fromProtocolTreeNode(resultNode))