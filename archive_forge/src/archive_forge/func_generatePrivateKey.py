import os
from .eckeypair import ECKeyPair
from ..invalidkeyexception import InvalidKeyException
import axolotl_curve25519 as _curve
@staticmethod
def generatePrivateKey():
    rand = os.urandom(32)
    return _curve.generatePrivateKey(rand)