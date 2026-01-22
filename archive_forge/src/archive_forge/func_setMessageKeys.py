from . import storageprotos_pb2 as storageprotos
from ..identitykeypair import IdentityKey, IdentityKeyPair
from ..ratchet.rootkey import RootKey
from ..kdf.hkdf import HKDF
from ..ecc.curve import Curve
from ..ecc.eckeypair import ECKeyPair
from ..ratchet.chainkey import ChainKey
from ..kdf.messagekeys import MessageKeys
def setMessageKeys(self, ECPublicKey_senderEphemeral, messageKeys):
    senderEphemeral = ECPublicKey_senderEphemeral
    chainAndIndex = self.getReceiverChain(senderEphemeral)
    chain = chainAndIndex[0]
    messageKeyStructure = chain.messageKeys.add()
    messageKeyStructure.cipherKey = messageKeys.getCipherKey()
    messageKeyStructure.macKey = messageKeys.getMacKey()
    messageKeyStructure.index = messageKeys.getCounter()
    messageKeyStructure.iv = messageKeys.getIv()
    self.sessionStructure.receiverChains[chainAndIndex[1]].CopyFrom(chain)