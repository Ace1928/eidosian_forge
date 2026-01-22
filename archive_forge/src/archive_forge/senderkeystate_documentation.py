from ...state import storageprotos_pb2 as storageprotos
from ..ratchet.senderchainkey import SenderChainKey
from ..ratchet.sendermessagekey import SenderMessageKey
from ...ecc.curve import Curve

        :type id: int
        :type iteration: int
        :type chainKey:  bytearray
        :type signatureKeyPublic:  ECPublicKey
        :type signatureKeyPrivate: ECPrivateKey
        :type signatureKeyPair: ECKeyPair
        :type senderKeyStateStructure: SenderKeyStateStructure
        