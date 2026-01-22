import os.path
import secrets
import ssl
import tempfile
import typing as t
def get_tls_server_end_point_data(certificate: t.Optional[bytes]) -> t.Optional[bytes]:
    """Get the TLS channel binding data.

    Gets the TLS channel binding data for tls-server-end-point used in
    Negotiate authentication.

    Args:
        tls_sock: The SSLSocket to get the binding data for.

    Returns:
        Optional[bytes]: The tls-server-end-point data used in the channel
        bindings application data value. Can return None if the cryptography
        isn't installed or it failed to get the certificate info.
    """
    if not HAS_CRYPTOGRAPHY or not certificate:
        return None
    cert = x509.load_der_x509_certificate(certificate)
    try:
        hash_algorithm = cert.signature_hash_algorithm
    except UnsupportedAlgorithm:
        hash_algorithm = None
    if not hash_algorithm or hash_algorithm.name in ['md5', 'sha1']:
        digest = hashes.Hash(hashes.SHA256())
    else:
        digest = hashes.Hash(hash_algorithm)
    digest.update(certificate)
    return b'tls-server-end-point:' + digest.finalize()