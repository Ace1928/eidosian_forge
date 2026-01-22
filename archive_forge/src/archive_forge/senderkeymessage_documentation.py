from .ciphertextmessage import CiphertextMessage
from ..util.byteutil import ByteUtil
from ..legacymessageexception import LegacyMessageException
from ..invalidmessageexception import InvalidMessageException
from ..invalidkeyexception import InvalidKeyException
from ..ecc.curve import Curve
from . import whisperprotos_pb2 as whisperprotos

        :type signatureKey: ECPrivateKey
        :type serialized: bytearray
        