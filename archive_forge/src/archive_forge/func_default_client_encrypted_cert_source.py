import six
from google.auth import exceptions
from google.auth.transport import _mtls_helper
def default_client_encrypted_cert_source(cert_path, key_path):
    """Get a callback which returns the default encrpyted client SSL credentials.

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
    """
    if not has_default_client_cert_source():
        raise exceptions.MutualTLSChannelError("Default client encrypted cert source doesn't exist")

    def callback():
        try:
            _, cert_bytes, key_bytes, passphrase_bytes = _mtls_helper.get_client_ssl_credentials(generate_encrypted_key=True)
            with open(cert_path, 'wb') as cert_file:
                cert_file.write(cert_bytes)
            with open(key_path, 'wb') as key_file:
                key_file.write(key_bytes)
        except (exceptions.ClientCertError, OSError) as caught_exc:
            new_exc = exceptions.MutualTLSChannelError(caught_exc)
            six.raise_from(new_exc, caught_exc)
        return (cert_path, key_path, passphrase_bytes)
    return callback