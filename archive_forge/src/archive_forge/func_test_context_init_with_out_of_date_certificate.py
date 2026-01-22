import datetime
import mock
import os
from cryptography.hazmat.backends import default_backend
from cryptography import x509
from cursive import certificate_utils
from cursive import exception
from cursive.tests import base
@mock.patch('cursive.certificate_utils.LOG')
@mock.patch('oslo_utils.timeutils.utcnow')
def test_context_init_with_out_of_date_certificate(self, mock_utcnow, mock_log):
    mock_utcnow.return_value = datetime.datetime(2100, 1, 1)
    certs = self.load_certificates(['self_signed_cert.pem', 'self_signed_cert.der'])
    cert_tuples = [('1', certs[0]), ('2', certs[1])]
    context = certificate_utils.CertificateVerificationContext(cert_tuples)
    self.assertEqual(0, len(context._signing_certificates))
    self.assertEqual(2, mock_log.warning.call_count)