import unittest
import time
import sys
from ..invalidkeyexception import InvalidKeyException
from ..sessionbuilder import SessionBuilder
from ..sessioncipher import SessionCipher
from ..ecc.curve import Curve
from ..protocol.ciphertextmessage import CiphertextMessage
from ..protocol.whispermessage import WhisperMessage
from ..protocol.prekeywhispermessage import PreKeyWhisperMessage
from ..state.prekeybundle import PreKeyBundle
from ..tests.inmemoryaxolotlstore import InMemoryAxolotlStore
from ..state.prekeyrecord import PreKeyRecord
from ..state.signedprekeyrecord import SignedPreKeyRecord
from ..tests.inmemoryidentitykeystore import InMemoryIdentityKeyStore
from ..protocol.keyexchangemessage import KeyExchangeMessage
from ..untrustedidentityexception import UntrustedIdentityException
def test_basicPreKeyV2(self):
    aliceStore = InMemoryAxolotlStore()
    aliceSessionBuilder = SessionBuilder(aliceStore, aliceStore, aliceStore, aliceStore, self.__class__.BOB_RECIPIENT_ID, 1)
    bobStore = InMemoryAxolotlStore()
    bobPreKeyPair = Curve.generateKeyPair()
    bobPreKey = PreKeyBundle(bobStore.getLocalRegistrationId(), 1, 31337, bobPreKeyPair.getPublicKey(), 0, None, None, bobStore.getIdentityKeyPair().getPublicKey())
    try:
        aliceSessionBuilder.processPreKeyBundle(bobPreKey)
        raise AssertionError('Should fail with missing unsigned prekey!')
    except InvalidKeyException:
        pass
    return