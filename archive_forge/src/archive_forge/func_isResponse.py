from .ciphertextmessage import CiphertextMessage
from ..util.byteutil import ByteUtil
from . import whisperprotos_pb2 as whisperprotos
from ..legacymessageexception import LegacyMessageException
from ..invalidversionexception import InvalidVersionException
from ..invalidmessageexception import InvalidMessageException
from ..invalidkeyexception import InvalidKeyException
from ..ecc.curve import Curve
from ..identitykey import IdentityKey
def isResponse(self):
    return self.flags & self.__class__.RESPONSE_FLAG != 0