import unittest
from ..state.sessionrecord import SessionRecord
from ..ecc.curve import Curve
from ..identitykeypair import IdentityKeyPair, IdentityKey
from ..ratchet.aliceaxolotlparameters import AliceAxolotlParameters
from ..ratchet.bobaxolotlparamaters import BobAxolotlParameters
from ..ratchet.ratchetingsession import RatchetingSession
from ..tests.inmemoryaxolotlstore import InMemoryAxolotlStore
from ..sessioncipher import SessionCipher
from ..protocol.whispermessage import WhisperMessage
def initializeSessionsV3(self, aliceSessionState, bobSessionState):
    aliceIdentityKeyPair = Curve.generateKeyPair()
    aliceIdentityKey = IdentityKeyPair(IdentityKey(aliceIdentityKeyPair.getPublicKey()), aliceIdentityKeyPair.getPrivateKey())
    aliceBaseKey = Curve.generateKeyPair()
    bobIdentityKeyPair = Curve.generateKeyPair()
    bobIdentityKey = IdentityKeyPair(IdentityKey(bobIdentityKeyPair.getPublicKey()), bobIdentityKeyPair.getPrivateKey())
    bobBaseKey = Curve.generateKeyPair()
    bobEphemeralKey = bobBaseKey
    aliceParameters = AliceAxolotlParameters.newBuilder().setOurBaseKey(aliceBaseKey).setOurIdentityKey(aliceIdentityKey).setTheirOneTimePreKey(None).setTheirRatchetKey(bobEphemeralKey.getPublicKey()).setTheirSignedPreKey(bobBaseKey.getPublicKey()).setTheirIdentityKey(bobIdentityKey.getPublicKey()).create()
    bobParameters = BobAxolotlParameters.newBuilder().setOurRatchetKey(bobEphemeralKey).setOurSignedPreKey(bobBaseKey).setOurOneTimePreKey(None).setOurIdentityKey(bobIdentityKey).setTheirIdentityKey(aliceIdentityKey.getPublicKey()).setTheirBaseKey(aliceBaseKey.getPublicKey()).create()
    RatchetingSession.initializeSessionAsAlice(aliceSessionState, aliceParameters)
    RatchetingSession.initializeSessionAsBob(bobSessionState, bobParameters)