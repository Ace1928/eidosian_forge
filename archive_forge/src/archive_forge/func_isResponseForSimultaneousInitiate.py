from .ciphertextmessage import CiphertextMessage
from ..util.byteutil import ByteUtil
from . import whisperprotos_pb2 as whisperprotos
from ..legacymessageexception import LegacyMessageException
from ..invalidversionexception import InvalidVersionException
from ..invalidmessageexception import InvalidMessageException
from ..invalidkeyexception import InvalidKeyException
from ..ecc.curve import Curve
from ..identitykey import IdentityKey
def isResponseForSimultaneousInitiate(self):
    return self.flags & self.__class__.SIMULTAENOUS_INITIATE_FLAG != 0