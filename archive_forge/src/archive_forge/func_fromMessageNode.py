from yowsup.structs import ProtocolTreeNode
from yowsup.layers.protocol_receipts.protocolentities import OutgoingReceiptProtocolEntity
from yowsup.layers.axolotl.protocolentities.iq_keys_get_result import ResultGetKeysIqProtocolEntity
@staticmethod
def fromMessageNode(message_node, local_registration_id):
    return RetryOutgoingReceiptProtocolEntity(message_node.getAttributeValue('id'), message_node.getAttributeValue('from'), local_registration_id, message_node.getAttributeValue('t'), participant=message_node.getAttributeValue('participant'))