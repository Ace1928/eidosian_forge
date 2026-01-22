from yowsup.layers.protocol_iq.protocolentities import IqProtocolEntity
from yowsup.structs import ProtocolTreeNode
def setValue(self, value):
    self.value = SetPrivacyIqProtocolEntity.checkValidValue(value)