from binascii import hexlify
from hashlib import md5, sha1
def calcHA1(pszAlg, pszUserName, pszRealm, pszPassword, pszNonce, pszCNonce, preHA1=None):
    """
    Compute H(A1) from RFC 2617.

    @param pszAlg: The name of the algorithm to use to calculate the digest.
        Currently supported are md5, md5-sess, and sha.
    @param pszUserName: The username
    @param pszRealm: The realm
    @param pszPassword: The password
    @param pszNonce: The nonce
    @param pszCNonce: The cnonce

    @param preHA1: If available this is a str containing a previously
       calculated H(A1) as a hex string.  If this is given then the values for
       pszUserName, pszRealm, and pszPassword must be L{None} and are ignored.
    """
    if preHA1 and (pszUserName or pszRealm or pszPassword):
        raise TypeError('preHA1 is incompatible with the pszUserName, pszRealm, and pszPassword arguments')
    if preHA1 is None:
        m = algorithms[pszAlg]()
        m.update(pszUserName)
        m.update(b':')
        m.update(pszRealm)
        m.update(b':')
        m.update(pszPassword)
        HA1 = hexlify(m.digest())
    else:
        HA1 = preHA1
    if pszAlg == b'md5-sess':
        m = algorithms[pszAlg]()
        m.update(HA1)
        m.update(b':')
        m.update(pszNonce)
        m.update(b':')
        m.update(pszCNonce)
        HA1 = hexlify(m.digest())
    return HA1