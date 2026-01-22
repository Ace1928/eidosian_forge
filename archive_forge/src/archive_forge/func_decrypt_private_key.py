import json
import logging
from os import path
import re
import subprocess
import six
from google.auth import exceptions
def decrypt_private_key(key, passphrase):
    """A helper function to decrypt the private key with the given passphrase.
    google-auth library doesn't support passphrase protected private key for
    mutual TLS channel. This helper function can be used to decrypt the
    passphrase protected private key in order to estalish mutual TLS channel.

    For example, if you have a function which produces client cert, passphrase
    protected private key and passphrase, you can convert it to a client cert
    callback function accepted by google-auth::

        from google.auth.transport import _mtls_helper

        def your_client_cert_function():
            return cert, encrypted_key, passphrase

        # callback accepted by google-auth for mutual TLS channel.
        def client_cert_callback():
            cert, encrypted_key, passphrase = your_client_cert_function()
            decrypted_key = _mtls_helper.decrypt_private_key(encrypted_key,
                passphrase)
            return cert, decrypted_key

    Args:
        key (bytes): The private key bytes in PEM format.
        passphrase (bytes): The passphrase bytes.

    Returns:
        bytes: The decrypted private key in PEM format.

    Raises:
        ImportError: If pyOpenSSL is not installed.
        OpenSSL.crypto.Error: If there is any problem decrypting the private key.
    """
    from OpenSSL import crypto
    pkey = crypto.load_privatekey(crypto.FILETYPE_PEM, key, passphrase=passphrase)
    return crypto.dump_privatekey(crypto.FILETYPE_PEM, pkey)