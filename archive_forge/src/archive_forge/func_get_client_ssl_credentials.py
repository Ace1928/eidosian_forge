import json
import logging
from os import path
import re
import subprocess
import six
from google.auth import exceptions
def get_client_ssl_credentials(generate_encrypted_key=False, context_aware_metadata_path=CONTEXT_AWARE_METADATA_PATH):
    """Returns the client side certificate, private key and passphrase.

    Args:
        generate_encrypted_key (bool): If set to True, encrypted private key
            and passphrase will be generated; otherwise, unencrypted private key
            will be generated and passphrase will be None.
        context_aware_metadata_path (str): The context_aware_metadata.json file path.

    Returns:
        Tuple[bool, bytes, bytes, bytes]:
            A boolean indicating if cert, key and passphrase are obtained, the
            cert bytes and key bytes both in PEM format, and passphrase bytes.

    Raises:
        google.auth.exceptions.ClientCertError: if problems occurs when getting
            the cert, key and passphrase.
    """
    metadata_path = _check_dca_metadata_path(context_aware_metadata_path)
    if metadata_path:
        metadata_json = _read_dca_metadata_file(metadata_path)
        if _CERT_PROVIDER_COMMAND not in metadata_json:
            raise exceptions.ClientCertError('Cert provider command is not found')
        command = metadata_json[_CERT_PROVIDER_COMMAND]
        if generate_encrypted_key and '--with_passphrase' not in command:
            command.append('--with_passphrase')
        cert, key, passphrase = _run_cert_provider_command(command, expect_encrypted_key=generate_encrypted_key)
        return (True, cert, key, passphrase)
    return (False, None, None, None)