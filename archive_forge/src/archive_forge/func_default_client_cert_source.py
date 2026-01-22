import six
from google.auth import exceptions
from google.auth.transport import _mtls_helper
def default_client_cert_source():
    """Get a callback which returns the default client SSL credentials.

    Returns:
        Callable[[], [bytes, bytes]]: A callback which returns the default
            client certificate bytes and private key bytes, both in PEM format.

    Raises:
        google.auth.exceptions.DefaultClientCertSourceError: If the default
            client SSL credentials don't exist or are malformed.
    """
    if not has_default_client_cert_source():
        raise exceptions.MutualTLSChannelError("Default client cert source doesn't exist")

    def callback():
        try:
            _, cert_bytes, key_bytes = _mtls_helper.get_client_cert_and_key()
        except (OSError, RuntimeError, ValueError) as caught_exc:
            new_exc = exceptions.MutualTLSChannelError(caught_exc)
            six.raise_from(new_exc, caught_exc)
        return (cert_bytes, key_bytes)
    return callback