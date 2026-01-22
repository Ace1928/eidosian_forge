import os.path
import secrets
import ssl
import tempfile
import typing as t
def load_client_certificate(context: ssl.SSLContext, certificate: str, key: t.Optional[str]=None, password: t.Optional[str]=None) -> None:
    """Loads a certificate to use with client authentication.

    Loads the supplied certificate that can be used for client authentication.
    This function is a wrapper around load_cert_chain and offers the ability to
    load a cert/key from a string or load a PFX formatted certificate with an
    optional password.

    The certificate argument can either be a string of the PEM encoded
    certificate and/or key. It can also be the path to a file of a PEM, DEF, or
    PKCS12 (pfx) certificate and/or key. The key argument can be used to
    specify the certificate key if it is not bundled with the certificate
    argument.

    Args:
        context: The SSLContext to load the cert info.
        certificate: The certificate as a string or filepath.
        key: The optional key as a string or filepath.
        password: The password that is used to decrypt the key or pfx file.
    """
    b_password = password.encode('utf-8', errors='surrogateescape') if password else None
    if os.path.isfile(certificate):
        with open(certificate, mode='rb') as fd:
            cert_data = fd.read()
        der_cert = _try_load_der_cert(cert_data)
        if der_cert:
            certificate = der_cert
        else:
            pfx_cert = _try_load_pfx_cert(cert_data, b_password)
            if pfx_cert:
                certificate, key, b_password = pfx_cert
    if key and os.path.isfile(key):
        with open(key, mode='rb') as fd:
            key_data = fd.read()
        der_key = _try_load_der_key(key_data, b_password)
        if der_key:
            key, b_password = der_key
    b_password = b_password or b''
    if certificate.startswith('-----') or (key and key.startswith('-----')):
        with tempfile.TemporaryDirectory() as tmpdir:
            if certificate.startswith('-----'):
                cert_path = os.path.join(tmpdir, 'cert.pem')
                with open(cert_path, mode='w') as fd:
                    fd.write(certificate)
                certificate = cert_path
            if key and key.startswith('-----'):
                key_path = os.path.join(tmpdir, 'key.pem')
                with open(key_path, mode='w') as fd:
                    fd.write(key)
                key = key_path
            context.load_cert_chain(certfile=certificate, keyfile=key, password=b_password)
        return
    context.load_cert_chain(certfile=certificate, keyfile=key, password=b_password)