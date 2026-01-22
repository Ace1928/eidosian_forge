import os
from .eckeypair import ECKeyPair
from ..invalidkeyexception import InvalidKeyException
import axolotl_curve25519 as _curve
@staticmethod
def generatePublicKey(privateKey):
    return _curve.generatePublicKey(privateKey)