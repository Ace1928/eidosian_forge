import hmac
import hashlib
from ..kdf.derivedmessagesecrets import DerivedMessageSecrets
from ..kdf.messagekeys import MessageKeys
def getKey(self):
    return self.key