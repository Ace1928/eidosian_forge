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
def test_get_certificate_key_manager_fail(self, mock_key_manager_API):
    bad_cert_uuid = 'fea14bc2-d75f-4ba5-bccc-b5c924ad0695'
    self.assertRaisesRegex(exception.SignatureVerificationError, 'Unable to retrieve certificate with ID: .*', signature_utils.get_certificate, None, bad_cert_uuid)