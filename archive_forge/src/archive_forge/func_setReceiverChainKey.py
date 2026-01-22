from . import storageprotos_pb2 as storageprotos
from ..identitykeypair import IdentityKey, IdentityKeyPair
from ..ratchet.rootkey import RootKey
from ..kdf.hkdf import HKDF
from ..ecc.curve import Curve
from ..ecc.eckeypair import ECKeyPair
from ..ratchet.chainkey import ChainKey
from ..kdf.messagekeys import MessageKeys
def setReceiverChainKey(self, ECPublicKey_senderEphemeral, chainKey):
    senderEphemeral = ECPublicKey_senderEphemeral
    chainAndIndex = self.getReceiverChain(senderEphemeral)
    chain = chainAndIndex[0]
    chain.chainKey.key = chainKey.getKey()
    chain.chainKey.index = chainKey.getIndex()
    self.sessionStructure.receiverChains[chainAndIndex[1]].CopyFrom(chain)