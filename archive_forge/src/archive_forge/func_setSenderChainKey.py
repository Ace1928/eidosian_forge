from ...state import storageprotos_pb2 as storageprotos
from ..ratchet.senderchainkey import SenderChainKey
from ..ratchet.sendermessagekey import SenderMessageKey
from ...ecc.curve import Curve
def setSenderChainKey(self, chainKey):
    self.senderKeyStateStructure.senderChainKey.iteration = chainKey.getIteration()
    self.senderKeyStateStructure.senderChainKey.seed = chainKey.getSeed()