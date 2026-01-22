import hmac
import hashlib
from ..kdf.derivedmessagesecrets import DerivedMessageSecrets
from ..kdf.messagekeys import MessageKeys
def getMessageKeys(self):
    inputKeyMaterial = self.getBaseMaterial(self.__class__.MESSAGE_KEY_SEED)
    keyMaterialBytes = self.kdf.deriveSecrets(inputKeyMaterial, bytearray('WhisperMessageKeys'.encode()), DerivedMessageSecrets.SIZE)
    keyMaterial = DerivedMessageSecrets(keyMaterialBytes)
    return MessageKeys(keyMaterial.getCipherKey(), keyMaterial.getMacKey(), keyMaterial.getIv(), self.index)