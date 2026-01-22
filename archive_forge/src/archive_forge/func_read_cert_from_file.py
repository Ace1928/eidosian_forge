import base64
import datetime
from os import remove
from os.path import join
from OpenSSL import crypto
import dateutil.parser
import pytz
import saml2.cryptography.pki
def read_cert_from_file(cert_file, cert_type='pem'):
    """Read a certificate from a file.

    If there are multiple certificates in the file, the first is returned.

    :param cert_file: The name of the file
    :param cert_type: The certificate type
    :return: A base64 encoded certificate as a string or the empty string
    """
    if not cert_file:
        return ''
    with open(cert_file, 'rb') as fp:
        data = fp.read()
    try:
        cert = saml2.cryptography.pki.load_x509_certificate(data, cert_type)
        pem_data = saml2.cryptography.pki.get_public_bytes_from_cert(cert)
    except Exception as e:
        raise CertificateError(e)
    pem_data_no_headers = ''.join(pem_data.splitlines()[1:-1])
    return pem_data_no_headers