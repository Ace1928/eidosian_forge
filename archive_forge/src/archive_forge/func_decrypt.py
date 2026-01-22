from ..invalidkeyidexception import InvalidKeyIdException
from ..invalidkeyexception import InvalidKeyException
from ..invalidmessageexception import InvalidMessageException
from ..duplicatemessagexception import DuplicateMessageException
from ..nosessionexception import NoSessionException
from ..protocol.senderkeymessage import SenderKeyMessage
from ..sessioncipher import AESCipher
from ..groups.state.senderkeystore import SenderKeyStore
def decrypt(self, senderKeyMessageBytes):
    """
        :type senderKeyMessageBytes: bytearray
        """
    try:
        record = self.senderKeyStore.loadSenderKey(self.senderKeyName)
        if record.isEmpty():
            raise NoSessionException('No sender key for: %s' % self.senderKeyName)
        senderKeyMessage = SenderKeyMessage(serialized=bytes(senderKeyMessageBytes))
        senderKeyState = record.getSenderKeyState(senderKeyMessage.getKeyId())
        senderKeyMessage.verifySignature(senderKeyState.getSigningKeyPublic())
        senderKey = self.getSenderKey(senderKeyState, senderKeyMessage.getIteration())
        plaintext = self.getPlainText(senderKey.getIv(), senderKey.getCipherKey(), senderKeyMessage.getCipherText())
        self.senderKeyStore.storeSenderKey(self.senderKeyName, record)
        return plaintext
    except (InvalidKeyException, InvalidKeyIdException) as e:
        raise InvalidMessageException(e)