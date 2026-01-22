import datetime
import mock
import os
from cryptography.hazmat.backends import default_backend
from cryptography import x509
from cursive import certificate_utils
from cursive import exception
from cursive.tests import base
@mock.patch('oslo_utils.timeutils.utcnow')
def test_context_verify(self, mock_utcnow):
    mock_utcnow.return_value = datetime.datetime(2017, 1, 1)
    certs = self.load_certificates(['self_signed_cert.pem', 'self_signed_cert.der'])
    cert_tuples = [('1', certs[0]), ('2', certs[1])]
    context = certificate_utils.CertificateVerificationContext(cert_tuples)
    cert = self.load_certificate('signed_cert.pem')
    context.update(cert)
    context.verify()
    context = certificate_utils.CertificateVerificationContext(cert_tuples)
    context.update(certs[0])
    context.verify()