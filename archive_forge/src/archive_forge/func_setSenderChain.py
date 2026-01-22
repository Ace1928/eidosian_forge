from . import storageprotos_pb2 as storageprotos
from ..identitykeypair import IdentityKey, IdentityKeyPair
from ..ratchet.rootkey import RootKey
from ..kdf.hkdf import HKDF
from ..ecc.curve import Curve
from ..ecc.eckeypair import ECKeyPair
from ..ratchet.chainkey import ChainKey
from ..kdf.messagekeys import MessageKeys
def setSenderChain(self, ECKeyPair_senderRatchetKeyPair, chainKey):
    senderRatchetKeyPair = ECKeyPair_senderRatchetKeyPair
    senderChain = storageprotos.SessionStructure.Chain()
    self.sessionStructure.senderChain.senderRatchetKey = senderRatchetKeyPair.getPublicKey().serialize()
    self.sessionStructure.senderChain.senderRatchetKeyPrivate = senderRatchetKeyPair.getPrivateKey().serialize()
    self.sessionStructure.senderChain.chainKey.key = chainKey.key
    self.sessionStructure.senderChain.chainKey.index = chainKey.index