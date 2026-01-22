import hmac
import hashlib
from .ciphertextmessage import CiphertextMessage
from ..util.byteutil import ByteUtil
from ..ecc.curve import Curve
from . import whisperprotos_pb2 as whisperprotos
from ..legacymessageexception import LegacyMessageException
from ..invalidmessageexception import InvalidMessageException
from ..invalidkeyexception import InvalidKeyException
def isLegacy(self, message):
    return message is not None and len(message) >= 1 and (ByteUtil.highBitsToInt(message[0]) <= CiphertextMessage.UNSUPPORTED_VERSION)