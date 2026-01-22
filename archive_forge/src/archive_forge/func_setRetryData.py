from yowsup.structs import ProtocolTreeNode
from yowsup.layers.protocol_receipts.protocolentities import IncomingReceiptProtocolEntity
from yowsup.layers.axolotl.protocolentities.iq_keys_get_result import ResultGetKeysIqProtocolEntity
def setRetryData(self, remoteRegistrationId, v, count, retryTimestamp):
    self.remoteRegistrationId = remoteRegistrationId
    self.v = int(v)
    self.count = int(count)
    self.retryTimestamp = int(retryTimestamp)