import datetime
import mock
import os
from cryptography.hazmat.backends import default_backend
from cryptography import x509
from cursive import certificate_utils
from cursive import exception
from cursive.tests import base
@mock.patch('cursive.signature_utils.get_certificate')
@mock.patch('oslo_utils.timeutils.utcnow')
def test_verify_valid_certificate_with_no_root(self, mock_utcnow, mock_get_cert):
    mock_utcnow.return_value = datetime.datetime(2017, 1, 1)
    certs = self.load_certificates(['signed_cert.pem'])
    mock_get_cert.side_effect = certs
    cert_uuid = '3'
    trusted_cert_uuids = []
    self.assertRaisesRegex(exception.SignatureVerificationError, 'Certificate chain building failed. Could not locate the signing certificate for the base certificate in the set of trusted certificates.', certificate_utils.verify_certificate, None, cert_uuid, trusted_cert_uuids)