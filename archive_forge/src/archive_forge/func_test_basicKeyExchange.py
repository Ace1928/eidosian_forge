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
def test_basicKeyExchange(self):
    aliceStore = InMemoryAxolotlStore()
    aliceSessionBuilder = SessionBuilder(aliceStore, aliceStore, aliceStore, aliceStore, self.__class__.BOB_RECIPIENT_ID, 1)
    bobStore = InMemoryAxolotlStore()
    bobSessionBuilder = SessionBuilder(bobStore, bobStore, bobStore, bobStore, self.__class__.ALICE_RECIPIENT_ID, 1)
    aliceKeyExchangeMessage = aliceSessionBuilder.processInitKeyExchangeMessage()
    self.assertTrue(aliceKeyExchangeMessage is not None)
    aliceKeyExchangeMessageBytes = aliceKeyExchangeMessage.serialize()
    bobKeyExchangeMessage = bobSessionBuilder.processKeyExchangeMessage(KeyExchangeMessage(serialized=aliceKeyExchangeMessageBytes))
    self.assertTrue(bobKeyExchangeMessage is not None)
    bobKeyExchangeMessageBytes = bobKeyExchangeMessage.serialize()
    response = aliceSessionBuilder.processKeyExchangeMessage(KeyExchangeMessage(serialized=bobKeyExchangeMessageBytes))
    self.assertTrue(response is None)
    self.assertTrue(aliceStore.containsSession(self.__class__.BOB_RECIPIENT_ID, 1))
    self.assertTrue(bobStore.containsSession(self.__class__.ALICE_RECIPIENT_ID, 1))
    self.runInteraction(aliceStore, bobStore)
    aliceStore = InMemoryAxolotlStore()
    aliceSessionBuilder = SessionBuilder(aliceStore, aliceStore, aliceStore, aliceStore, self.__class__.BOB_RECIPIENT_ID, 1)
    aliceKeyExchangeMessage = aliceSessionBuilder.processInitKeyExchangeMessage()
    try:
        bobKeyExchangeMessage = bobSessionBuilder.processKeyExchangeMessage(aliceKeyExchangeMessage)
        raise AssertionError("This identity shouldn't be trusted!")
    except UntrustedIdentityException:
        bobStore.saveIdentity(self.__class__.ALICE_RECIPIENT_ID, aliceKeyExchangeMessage.getIdentityKey())
    bobKeyExchangeMessage = bobSessionBuilder.processKeyExchangeMessage(aliceKeyExchangeMessage)
    self.assertTrue(aliceSessionBuilder.processKeyExchangeMessage(bobKeyExchangeMessage) == None)
    self.runInteraction(aliceStore, bobStore)