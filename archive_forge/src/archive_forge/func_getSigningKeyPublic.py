from ...state import storageprotos_pb2 as storageprotos
from ..ratchet.senderchainkey import SenderChainKey
from ..ratchet.sendermessagekey import SenderMessageKey
from ...ecc.curve import Curve
def getSigningKeyPublic(self):
    return Curve.decodePoint(bytearray(self.senderKeyStateStructure.senderSigningKey.public), 0)