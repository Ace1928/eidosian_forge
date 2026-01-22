from ..ecc.curve import Curve
from .bobaxolotlparamaters import BobAxolotlParameters
from .aliceaxolotlparameters import AliceAxolotlParameters
from ..kdf.hkdfv3 import HKDFv3
from ..util.byteutil import ByteUtil
from .rootkey import RootKey
from .chainkey import ChainKey
from ..protocol.ciphertextmessage import CiphertextMessage
@staticmethod
def initializeSessionAsAlice(sessionState, parameters):
    """
        :type sessionState: SessionState
        :type parameters: AliceAxolotlParameters
        """
    sessionState.setSessionVersion(CiphertextMessage.CURRENT_VERSION)
    sessionState.setRemoteIdentityKey(parameters.getTheirIdentityKey())
    sessionState.setLocalIdentityKey(parameters.getOurIdentityKey().getPublicKey())
    sendingRatchetKey = Curve.generateKeyPair()
    secrets = bytearray()
    secrets.extend(RatchetingSession.getDiscontinuityBytes())
    secrets.extend(Curve.calculateAgreement(parameters.getTheirSignedPreKey(), parameters.getOurIdentityKey().getPrivateKey()))
    secrets.extend(Curve.calculateAgreement(parameters.getTheirIdentityKey().getPublicKey(), parameters.getOurBaseKey().getPrivateKey()))
    secrets.extend(Curve.calculateAgreement(parameters.getTheirSignedPreKey(), parameters.getOurBaseKey().getPrivateKey()))
    if parameters.getTheirOneTimePreKey() is not None:
        secrets.extend(Curve.calculateAgreement(parameters.getTheirOneTimePreKey(), parameters.getOurBaseKey().getPrivateKey()))
    derivedKeys = RatchetingSession.calculateDerivedKeys(secrets)
    sendingChain = derivedKeys.getRootKey().createChain(parameters.getTheirRatchetKey(), sendingRatchetKey)
    sessionState.addReceiverChain(parameters.getTheirRatchetKey(), derivedKeys.getChainKey())
    sessionState.setSenderChain(sendingRatchetKey, sendingChain[1])
    sessionState.setRootKey(sendingChain[0])