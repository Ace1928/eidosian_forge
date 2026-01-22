from __future__ import absolute_import, division, print_function
import os
from base64 import b64encode, b64decode
from getpass import getuser
from socket import gethostname
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
@staticmethod
def encode_openssh_privatekey(asym_keypair, key_format):
    """Returns an OpenSSH encoded private key for a given keypair

           :asym_keypair: Asymmetric_Keypair from the private key is extracted
           :key_format: Format of the encoded private key.
        """
    if key_format == 'SSH':
        if not HAS_OPENSSH_PRIVATE_FORMAT:
            privatekey_format = serialization.PrivateFormat.PKCS8
        else:
            privatekey_format = serialization.PrivateFormat.OpenSSH
    elif key_format == 'PKCS8':
        privatekey_format = serialization.PrivateFormat.PKCS8
    elif key_format == 'PKCS1':
        if asym_keypair.key_type == 'ed25519':
            raise InvalidKeyFormatError('ed25519 keys cannot be represented in PKCS1 format')
        privatekey_format = serialization.PrivateFormat.TraditionalOpenSSL
    else:
        raise InvalidKeyFormatError('The accepted private key formats are SSH, PKCS8, and PKCS1')
    encoded_privatekey = asym_keypair.private_key.private_bytes(encoding=serialization.Encoding.PEM, format=privatekey_format, encryption_algorithm=asym_keypair.encryption_algorithm)
    return encoded_privatekey