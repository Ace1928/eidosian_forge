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
def processV3(self, sessionRecord, message):
    """
        :param sessionRecord:
        :param message:
        :type message: PreKeyWhisperMessage
        :return:
        """
    if sessionRecord.hasSessionState(message.getMessageVersion(), message.getBaseKey().serialize()):
        logger.warn("We've already setup a session for this V3 message, letting bundled message fall through...")
        return None
    ourSignedPreKey = self.signedPreKeyStore.loadSignedPreKey(message.getSignedPreKeyId()).getKeyPair()
    parameters = BobAxolotlParameters.newBuilder()
    parameters.setTheirBaseKey(message.getBaseKey()).setTheirIdentityKey(message.getIdentityKey()).setOurIdentityKey(self.identityKeyStore.getIdentityKeyPair()).setOurSignedPreKey(ourSignedPreKey).setOurRatchetKey(ourSignedPreKey)
    if message.getPreKeyId() is not None:
        parameters.setOurOneTimePreKey(self.preKeyStore.loadPreKey(message.getPreKeyId()).getKeyPair())
    else:
        parameters.setOurOneTimePreKey(None)
    if not sessionRecord.isFresh():
        sessionRecord.archiveCurrentState()
    RatchetingSession.initializeSessionAsBob(sessionRecord.getSessionState(), parameters.create())
    sessionRecord.getSessionState().setLocalRegistrationId(self.identityKeyStore.getLocalRegistrationId())
    sessionRecord.getSessionState().setRemoteRegistrationId(message.getRegistrationId())
    sessionRecord.getSessionState().setAliceBaseKey(message.getBaseKey().serialize())
    if message.getPreKeyId() is not None and message.getPreKeyId() != Medium.MAX_VALUE:
        return message.getPreKeyId()
    else:
        return None