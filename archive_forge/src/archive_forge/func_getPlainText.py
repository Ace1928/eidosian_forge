from ..invalidkeyidexception import InvalidKeyIdException
from ..invalidkeyexception import InvalidKeyException
from ..invalidmessageexception import InvalidMessageException
from ..duplicatemessagexception import DuplicateMessageException
from ..nosessionexception import NoSessionException
from ..protocol.senderkeymessage import SenderKeyMessage
from ..sessioncipher import AESCipher
from ..groups.state.senderkeystore import SenderKeyStore
def getPlainText(self, iv, key, ciphertext):
    """
        :type iv: bytearray
        :type key: bytearray
        :type ciphertext: bytearray
        """
    try:
        cipher = AESCipher(key, iv)
        plaintext = cipher.decrypt(ciphertext)
        return plaintext
    except Exception as e:
        raise InvalidMessageException(e)