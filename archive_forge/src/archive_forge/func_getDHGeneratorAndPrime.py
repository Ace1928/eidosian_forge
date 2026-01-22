from hashlib import sha1, sha256, sha384, sha512
from zope.interface import Attribute, Interface, implementer
from twisted.conch import error
def getDHGeneratorAndPrime(kexAlgorithm):
    """
    Get the generator and the prime to use in key exchange.

    @param kexAlgorithm: The key exchange algorithm name.
    @type kexAlgorithm: L{bytes}

    @return: A L{tuple} containing L{int} generator and L{int} prime.
    @rtype: L{tuple}
    """
    kex = getKex(kexAlgorithm)
    return (kex.generator, kex.prime)