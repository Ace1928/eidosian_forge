from ..ecc.curve import Curve
from .bobaxolotlparamaters import BobAxolotlParameters
from .aliceaxolotlparameters import AliceAxolotlParameters
from ..kdf.hkdfv3 import HKDFv3
from ..util.byteutil import ByteUtil
from .rootkey import RootKey
from .chainkey import ChainKey
from ..protocol.ciphertextmessage import CiphertextMessage
@staticmethod
def calculateDerivedKeys(masterSecret):
    kdf = HKDFv3()
    derivedSecretBytes = kdf.deriveSecrets(masterSecret, bytearray('WhisperText'.encode()), 64)
    derivedSecrets = ByteUtil.split(derivedSecretBytes, 32, 32)
    return RatchetingSession.DerivedKeys(RootKey(kdf, derivedSecrets[0]), ChainKey(kdf, derivedSecrets[1], 0))