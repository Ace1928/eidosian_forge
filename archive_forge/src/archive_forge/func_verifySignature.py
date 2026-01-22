from .ciphertextmessage import CiphertextMessage
from ..util.byteutil import ByteUtil
from ..legacymessageexception import LegacyMessageException
from ..invalidmessageexception import InvalidMessageException
from ..invalidkeyexception import InvalidKeyException
from ..ecc.curve import Curve
from . import whisperprotos_pb2 as whisperprotos
def verifySignature(self, signatureKey):
    """
        :type signatureKey: ECPublicKey
        """
    try:
        parts = ByteUtil.split(self.serialized, len(self.serialized) - self.__class__.SIGNATURE_LENGTH, self.__class__.SIGNATURE_LENGTH)
        if not Curve.verifySignature(signatureKey, parts[0], parts[1]):
            raise InvalidMessageException('Invalid signature!')
    except InvalidKeyException as e:
        raise InvalidMessageException(e)