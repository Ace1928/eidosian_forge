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
def test_basicPreKeyV3(self):
    aliceStore = InMemoryAxolotlStore()
    aliceSessionBuilder = SessionBuilder(aliceStore, aliceStore, aliceStore, aliceStore, self.__class__.BOB_RECIPIENT_ID, 1)
    bobStore = InMemoryAxolotlStore()
    bobPreKeyPair = Curve.generateKeyPair()
    bobSignedPreKeyPair = Curve.generateKeyPair()
    bobSignedPreKeySignature = Curve.calculateSignature(bobStore.getIdentityKeyPair().getPrivateKey(), bobSignedPreKeyPair.getPublicKey().serialize())
    bobPreKey = PreKeyBundle(bobStore.getLocalRegistrationId(), 1, 31337, bobPreKeyPair.getPublicKey(), 22, bobSignedPreKeyPair.getPublicKey(), bobSignedPreKeySignature, bobStore.getIdentityKeyPair().getPublicKey())
    aliceSessionBuilder.processPreKeyBundle(bobPreKey)
    self.assertTrue(aliceStore.containsSession(self.__class__.BOB_RECIPIENT_ID, 1))
    self.assertTrue(aliceStore.loadSession(self.__class__.BOB_RECIPIENT_ID, 1).getSessionState().getSessionVersion() == 3)
    originalMessage = b"L'homme est condamne a etre libre"
    aliceSessionCipher = SessionCipher(aliceStore, aliceStore, aliceStore, aliceStore, self.__class__.BOB_RECIPIENT_ID, 1)
    outgoingMessage = aliceSessionCipher.encrypt(originalMessage)
    self.assertTrue(outgoingMessage.getType() == CiphertextMessage.PREKEY_TYPE)
    incomingMessage = PreKeyWhisperMessage(serialized=outgoingMessage.serialize())
    bobStore.storePreKey(31337, PreKeyRecord(bobPreKey.getPreKeyId(), bobPreKeyPair))
    bobStore.storeSignedPreKey(22, SignedPreKeyRecord(22, int(time.time() * 1000), bobSignedPreKeyPair, bobSignedPreKeySignature))
    bobSessionCipher = SessionCipher(bobStore, bobStore, bobStore, bobStore, self.__class__.ALICE_RECIPIENT_ID, 1)
    plaintext = bobSessionCipher.decryptPkmsg(incomingMessage)
    self.assertEqual(originalMessage, plaintext)
    self.assertTrue(bobStore.containsSession(self.__class__.ALICE_RECIPIENT_ID, 1))
    self.assertTrue(bobStore.loadSession(self.__class__.ALICE_RECIPIENT_ID, 1).getSessionState().getSessionVersion() == 3)
    self.assertTrue(bobStore.loadSession(self.__class__.ALICE_RECIPIENT_ID, 1).getSessionState().getAliceBaseKey() is not None)
    self.assertEqual(originalMessage, plaintext)
    bobOutgoingMessage = bobSessionCipher.encrypt(originalMessage)
    self.assertTrue(bobOutgoingMessage.getType() == CiphertextMessage.WHISPER_TYPE)
    alicePlaintext = aliceSessionCipher.decryptMsg(WhisperMessage(serialized=bobOutgoingMessage.serialize()))
    self.assertEqual(alicePlaintext, originalMessage)
    self.runInteraction(aliceStore, bobStore)
    aliceStore = InMemoryAxolotlStore()
    aliceSessionBuilder = SessionBuilder(aliceStore, aliceStore, aliceStore, aliceStore, self.__class__.BOB_RECIPIENT_ID, 1)
    aliceSessionCipher = SessionCipher(aliceStore, aliceStore, aliceStore, aliceStore, self.__class__.BOB_RECIPIENT_ID, 1)
    bobPreKeyPair = Curve.generateKeyPair()
    bobSignedPreKeyPair = Curve.generateKeyPair()
    bobSignedPreKeySignature = Curve.calculateSignature(bobStore.getIdentityKeyPair().getPrivateKey(), bobSignedPreKeyPair.getPublicKey().serialize())
    bobPreKey = PreKeyBundle(bobStore.getLocalRegistrationId(), 1, 31338, bobPreKeyPair.getPublicKey(), 23, bobSignedPreKeyPair.getPublicKey(), bobSignedPreKeySignature, bobStore.getIdentityKeyPair().getPublicKey())
    bobStore.storePreKey(31338, PreKeyRecord(bobPreKey.getPreKeyId(), bobPreKeyPair))
    bobStore.storeSignedPreKey(23, SignedPreKeyRecord(23, int(time.time() * 1000), bobSignedPreKeyPair, bobSignedPreKeySignature))
    aliceSessionBuilder.processPreKeyBundle(bobPreKey)
    outgoingMessage = aliceSessionCipher.encrypt(originalMessage)
    try:
        plaintext = bobSessionCipher.decryptPkmsg(PreKeyWhisperMessage(serialized=outgoingMessage))
        raise AssertionError("shouldn't be trusted!")
    except Exception:
        bobStore.saveIdentity(self.__class__.ALICE_RECIPIENT_ID, PreKeyWhisperMessage(serialized=outgoingMessage.serialize()).getIdentityKey())
    plaintext = bobSessionCipher.decryptPkmsg(PreKeyWhisperMessage(serialized=outgoingMessage.serialize()))
    self.assertEqual(plaintext, originalMessage)
    bobPreKey = PreKeyBundle(bobStore.getLocalRegistrationId(), 1, 31337, Curve.generateKeyPair().getPublicKey(), 23, bobSignedPreKeyPair.getPublicKey(), bobSignedPreKeySignature, aliceStore.getIdentityKeyPair().getPublicKey())
    try:
        aliceSessionBuilder.process(bobPreKey)
        raise AssertionError("shouldn't be trusted!")
    except Exception:
        pass