from hashlib import sha1, sha256, sha384, sha512
from zope.interface import Attribute, Interface, implementer
from twisted.conch import error
def isEllipticCurve(kexAlgorithm):
    """
    Returns C{True} if C{kexAlgorithm} is an elliptic curve.

    @param kexAlgorithm: The key exchange algorithm name.
    @type kexAlgorithm: C{str}

    @return: C{True} if C{kexAlgorithm} is an elliptic curve,
        otherwise C{False}.
    @rtype: C{bool}
    """
    return _IEllipticCurveExchangeKexAlgorithm.providedBy(getKex(kexAlgorithm))