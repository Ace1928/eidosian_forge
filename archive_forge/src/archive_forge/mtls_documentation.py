import six
from google.auth import exceptions
from google.auth.transport import _mtls_helper
Get a callback which returns the default encrpyted client SSL credentials.

    Args:
        cert_path (str): The cert file path. The default client certificate will
            be written to this file when the returned callback is called.
        key_path (str): The key file path. The default encrypted client key will
            be written to this file when the returned callback is called.

    Returns:
        Callable[[], [str, str, bytes]]: A callback which generates the default
            client certificate, encrpyted private key and passphrase. It writes
            the certificate and private key into the cert_path and key_path, and
            returns the cert_path, key_path and passphrase bytes.

    Raises:
        google.auth.exceptions.DefaultClientCertSourceError: If any problem
            occurs when loading or saving the client certificate and key.
    