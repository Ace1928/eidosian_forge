import logging
from .ecc.curve import Curve
from .ratchet.aliceaxolotlparameters import AliceAxolotlParameters
from .ratchet.bobaxolotlparamaters import BobAxolotlParameters
from .ratchet.symmetricaxolotlparameters import SymmetricAxolotlParameters
from .ratchet.ratchetingsession import RatchetingSession
from .invalidkeyexception import InvalidKeyException
from .invalidkeyidexception import InvalidKeyIdException
from .untrustedidentityexception import UntrustedIdentityException
from .protocol.keyexchangemessage import KeyExchangeMessage
from .protocol.ciphertextmessage import CiphertextMessage
from .statekeyexchangeexception import StaleKeyExchangeException
from .util.medium import Medium
from .util.keyhelper import KeyHelper
def processPreKeyBundle(self, preKey):
    """
        :type preKey: PreKeyBundle
        """
    if not self.identityKeyStore.isTrustedIdentity(self.recipientId, preKey.getIdentityKey()):
        raise UntrustedIdentityException(self.recipientId, preKey.getIdentityKey())
    if preKey.getSignedPreKey() is not None and (not Curve.verifySignature(preKey.getIdentityKey().getPublicKey(), preKey.getSignedPreKey().serialize(), preKey.getSignedPreKeySignature())):
        raise InvalidKeyException('Invalid signature on device key!')
    if preKey.getSignedPreKey() is None:
        raise InvalidKeyException('No signed prekey!!')
    sessionRecord = self.sessionStore.loadSession(self.recipientId, self.deviceId)
    ourBaseKey = Curve.generateKeyPair()
    theirSignedPreKey = preKey.getSignedPreKey()
    theirOneTimePreKey = preKey.getPreKey()
    theirOneTimePreKeyId = preKey.getPreKeyId() if theirOneTimePreKey is not None else None
    parameters = AliceAxolotlParameters.newBuilder()
    parameters.setOurBaseKey(ourBaseKey).setOurIdentityKey(self.identityKeyStore.getIdentityKeyPair()).setTheirIdentityKey(preKey.getIdentityKey()).setTheirSignedPreKey(theirSignedPreKey).setTheirRatchetKey(theirSignedPreKey).setTheirOneTimePreKey(theirOneTimePreKey)
    if not sessionRecord.isFresh():
        sessionRecord.archiveCurrentState()
    RatchetingSession.initializeSessionAsAlice(sessionRecord.getSessionState(), parameters.create())
    sessionRecord.getSessionState().setUnacknowledgedPreKeyMessage(theirOneTimePreKeyId, preKey.getSignedPreKeyId(), ourBaseKey.getPublicKey())
    sessionRecord.getSessionState().setLocalRegistrationId(self.identityKeyStore.getLocalRegistrationId())
    sessionRecord.getSessionState().setRemoteRegistrationId(preKey.getRegistrationId())
    sessionRecord.getSessionState().setAliceBaseKey(ourBaseKey.getPublicKey().serialize())
    self.sessionStore.storeSession(self.recipientId, self.deviceId, sessionRecord)
    self.identityKeyStore.saveIdentity(self.recipientId, preKey.getIdentityKey())