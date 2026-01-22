import datetime
import mock
import os
from cryptography.hazmat.backends import default_backend
from cryptography import x509
from cursive import certificate_utils
from cursive import exception
from cursive.tests import base
@mock.patch('oslo_utils.timeutils.utcnow')
def test_is_within_valid_dates(self, mock_utcnow):
    cert = self.load_certificate('self_signed_cert.pem')
    mock_utcnow.return_value = datetime.datetime(2017, 1, 1)
    result = certificate_utils.is_within_valid_dates(cert)
    self.assertEqual(True, result)