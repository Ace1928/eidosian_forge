from binascii import hexlify
from hashlib import md5, sha1
def calcHA2(algo, pszMethod, pszDigestUri, pszQop, pszHEntity):
    """
    Compute H(A2) from RFC 2617.

    @param algo: The name of the algorithm to use to calculate the digest.
        Currently supported are md5, md5-sess, and sha.
    @param pszMethod: The request method.
    @param pszDigestUri: The request URI.
    @param pszQop: The Quality-of-Protection value.
    @param pszHEntity: The hash of the entity body or L{None} if C{pszQop} is
        not C{'auth-int'}.
    @return: The hash of the A2 value for the calculation of the response
        digest.
    """
    m = algorithms[algo]()
    m.update(pszMethod)
    m.update(b':')
    m.update(pszDigestUri)
    if pszQop == b'auth-int':
        m.update(b':')
        m.update(pszHEntity)
    return hexlify(m.digest())