import base64
import datetime
import mock
from castellan.common.exception import KeyManagerError
from castellan.common.exception import ManagedObjectNotFoundError
import cryptography.exceptions as crypto_exceptions
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import dsa
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric import rsa
from oslo_utils import timeutils
from cursive import exception
from cursive import signature_utils
from cursive.tests import base
@mock.patch('castellan.key_manager.API', return_value=FakeKeyManager())
def test_get_certificate_invalid_format(self, mock_API):
    cert_uuid = 'invalid_format_cert'
    self.assertRaisesRegex(exception.SignatureVerificationError, 'Invalid certificate format: .*', signature_utils.get_certificate, None, cert_uuid)