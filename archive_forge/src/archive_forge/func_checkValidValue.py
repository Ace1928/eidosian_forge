from yowsup.layers.protocol_iq.protocolentities import IqProtocolEntity
from yowsup.structs import ProtocolTreeNode
@staticmethod
def checkValidValue(value):
    if not value in SetPrivacyIqProtocolEntity.VALUES:
        raise Exception("Value should be in: '" + "', '".join(SetPrivacyIqProtocolEntity.VALUES) + "' but is '" + value + "'")
    return value