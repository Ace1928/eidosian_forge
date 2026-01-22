import sys
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
from .ecc.curve import Curve
from .sessionbuilder import SessionBuilder
from .state.sessionstate import SessionState
from .protocol.whispermessage import WhisperMessage
from .protocol.prekeywhispermessage import PreKeyWhisperMessage
from .nosessionexception import NoSessionException
from .invalidmessageexception import InvalidMessageException
from .duplicatemessagexception import DuplicateMessageException
import  logging
def getCiphertext(self, version, messageKeys, plainText):
    """
        :type version: int
        :type messageKeys: MessageKeys
        :type  plainText: bytearray
        """
    cipher = self.getCipher(messageKeys.getCipherKey(), messageKeys.getIv())
    return cipher.encrypt(plainText)