import logging
import warnings
from rsa._compat import range
import rsa.prime
import rsa.pem
import rsa.common
import rsa.randnum
import rsa.core
@classmethod
def load_pkcs1_openssl_der(cls, keyfile):
    """Loads a PKCS#1 DER-encoded public key file from OpenSSL.

        :param keyfile: contents of a DER-encoded file that contains the public
            key, from OpenSSL.
        :return: a PublicKey object
        :rtype: bytes

        """
    from rsa.asn1 import OpenSSLPubKey
    from pyasn1.codec.der import decoder
    from pyasn1.type import univ
    keyinfo, _ = decoder.decode(keyfile, asn1Spec=OpenSSLPubKey())
    if keyinfo['header']['oid'] != univ.ObjectIdentifier('1.2.840.113549.1.1.1'):
        raise TypeError('This is not a DER-encoded OpenSSL-compatible public key')
    return cls._load_pkcs1_der(keyinfo['key'][1:])