import os
from .eckeypair import ECKeyPair
from ..invalidkeyexception import InvalidKeyException
import axolotl_curve25519 as _curve
@staticmethod
def calculateAgreement(publicKey, privateKey):
    """
        :type publicKey: ECPublicKey
        :type privateKey: ECPrivateKey
        """
    if publicKey.getType() != privateKey.getType():
        raise InvalidKeyException('Public and private keys must be of the same type!')
    if publicKey.getType() == Curve.DJB_TYPE:
        return _curve.calculateAgreement(privateKey.getPrivateKey(), publicKey.getPublicKey())
    else:
        raise InvalidKeyException('Unknown type: %s' % publicKey.getType())