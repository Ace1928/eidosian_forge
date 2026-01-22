from ...state import storageprotos_pb2 as storageprotos
from ..ratchet.senderchainkey import SenderChainKey
from ..ratchet.sendermessagekey import SenderMessageKey
from ...ecc.curve import Curve
def getSenderChainKey(self):
    return SenderChainKey(self.senderKeyStateStructure.senderChainKey.iteration, bytearray(self.senderKeyStateStructure.senderChainKey.seed))