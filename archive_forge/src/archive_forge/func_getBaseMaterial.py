import hmac
import hashlib
from ..kdf.derivedmessagesecrets import DerivedMessageSecrets
from ..kdf.messagekeys import MessageKeys
def getBaseMaterial(self, seedBytes):
    mac = hmac.new(bytes(self.key), digestmod=hashlib.sha256)
    mac.update(bytes(seedBytes))
    return mac.digest()