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
def processResponse(self, keyExchangeMessage):
    sessionRecord = self.sessionStore.loadSession(self.recipientId, self.deviceId)
    sessionState = sessionRecord.getSessionState()
    hasPendingKeyExchange = sessionState.hasPendingKeyExchange()
    isSimultaneousInitiateResponse = keyExchangeMessage.isResponseForSimultaneousInitiate()
    if not hasPendingKeyExchange or sessionState.getPendingKeyExchangeSequence() != keyExchangeMessage.getSequence():
        logger.warn('No matching sequence for response. Is simultaneous initiate response: %s' % isSimultaneousInitiateResponse)
        if not isSimultaneousInitiateResponse:
            raise StaleKeyExchangeException()
        else:
            return
    parameters = SymmetricAxolotlParameters.newBuilder()
    parameters.setOurBaseKey(sessionRecord.getSessionState().getPendingKeyExchangeBaseKey()).setOurRatchetKey(sessionRecord.getSessionState().getPendingKeyExchangeRatchetKey()).setOurIdentityKey(sessionRecord.getSessionState().getPendingKeyExchangeIdentityKey()).setTheirBaseKey(keyExchangeMessage.getBaseKey()).setTheirRatchetKey(keyExchangeMessage.getRatchetKey()).setTheirIdentityKey(keyExchangeMessage.getIdentityKey())
    if not sessionRecord.isFresh():
        sessionRecord.archiveCurrentState()
    RatchetingSession.initializeSession(sessionRecord.getSessionState(), parameters.create())
    if not Curve.verifySignature(keyExchangeMessage.getIdentityKey().getPublicKey(), keyExchangeMessage.getBaseKey().serialize(), keyExchangeMessage.getBaseKeySignature()):
        raise InvalidKeyException("Base key signature doesn't match!")
    self.sessionStore.storeSession(self.recipientId, self.deviceId, sessionRecord)
    self.identityKeyStore.saveIdentity(self.recipientId, keyExchangeMessage.getIdentityKey())