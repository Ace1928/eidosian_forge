import hashlib
import hmac
from .sendermessagekey import SenderMessageKey
def getSenderMessageKey(self):
    return SenderMessageKey(self.iteration, self.getDerivative(self.__class__.MESSAGE_KEY_SEED, self.chainKey))