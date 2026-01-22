import os.path
import secrets
import ssl
import tempfile
import typing as t
def load_trust_certificate(context: ssl.SSLContext, certificate: str) -> None:
    """Loads a certificate as a trusted CA.

    Loads the supplied certificate info into the SSLContext to trust. The
    certificate can be in the following 3 forms:

        file path: The path to a PEM or DER encoded certificate
        dir path: The path to a directory containing multiple CA PEM files in
            a specific OpenSSL format (see c_rehash in OpenSSL).
        string: A PEM encoded certificate as a string.

    Args:
        context: The SSLContext to load the cert into.
        certificate: The certificate info to trust.
    """
    if os.path.exists(certificate):
        if os.path.isdir(certificate):
            context.load_verify_locations(capath=certificate)
        else:
            with open(certificate, mode='rb') as fd:
                data = fd.read()
            if data.startswith(b'-----BEGIN CERTIFICATE-----'):
                context.load_verify_locations(cafile=certificate)
            else:
                context.load_verify_locations(cadata=data)
    else:
        context.load_verify_locations(cadata=certificate)