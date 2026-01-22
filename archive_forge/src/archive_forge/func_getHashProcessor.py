from hashlib import sha1, sha256, sha384, sha512
from zope.interface import Attribute, Interface, implementer
from twisted.conch import error
def getHashProcessor(kexAlgorithm):
    """
    Get the hash algorithm callable to use in key exchange.

    @param kexAlgorithm: The key exchange algorithm name.
    @type kexAlgorithm: L{bytes}

    @return: A callable hash algorithm constructor (e.g. C{hashlib.sha256}).
    @rtype: C{callable}
    """
    kex = getKex(kexAlgorithm)
    return kex.hashProcessor