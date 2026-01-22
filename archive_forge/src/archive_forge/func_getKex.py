from hashlib import sha1, sha256, sha384, sha512
from zope.interface import Attribute, Interface, implementer
from twisted.conch import error
def getKex(kexAlgorithm):
    """
    Get a description of a named key exchange algorithm.

    @param kexAlgorithm: The key exchange algorithm name.
    @type kexAlgorithm: L{bytes}

    @return: A description of the key exchange algorithm named by
        C{kexAlgorithm}.
    @rtype: L{_IKexAlgorithm}

    @raises ConchError: if the key exchange algorithm is not found.
    """
    if kexAlgorithm not in _kexAlgorithms:
        raise error.ConchError(f'Unsupported key exchange algorithm: {kexAlgorithm}')
    return _kexAlgorithms[kexAlgorithm]