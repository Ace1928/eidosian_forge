from . import storageprotos_pb2 as storageprotos
from ..identitykeypair import IdentityKey, IdentityKeyPair
from ..ratchet.rootkey import RootKey
from ..kdf.hkdf import HKDF
from ..ecc.curve import Curve
from ..ecc.eckeypair import ECKeyPair
from ..ratchet.chainkey import ChainKey
from ..kdf.messagekeys import MessageKeys
def hasMessageKeys(self, ECPublickKey_senderEphemeral, counter):
    senderEphemeral = ECPublickKey_senderEphemeral
    chainAndIndex = self.getReceiverChain(senderEphemeral)
    chain = chainAndIndex[0]
    if chain is None:
        return False
    messageKeyList = chain.messageKeys
    for messageKey in messageKeyList:
        if messageKey.index == counter:
            return True
    return False