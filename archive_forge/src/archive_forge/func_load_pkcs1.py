import logging
import warnings
from rsa._compat import range
import rsa.prime
import rsa.pem
import rsa.common
import rsa.randnum
import rsa.core
@classmethod
def load_pkcs1(cls, keyfile, format='PEM'):
    """Loads a key in PKCS#1 DER or PEM format.

        :param keyfile: contents of a DER- or PEM-encoded file that contains
            the key.
        :type keyfile: bytes
        :param format: the format of the file to load; 'PEM' or 'DER'
        :type format: str

        :return: the loaded key
        :rtype: AbstractKey
        """
    methods = {'PEM': cls._load_pkcs1_pem, 'DER': cls._load_pkcs1_der}
    method = cls._assert_format_exists(format, methods)
    return method(keyfile)