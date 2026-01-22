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
@mock.patch('cursive.signature_utils.get_certificate')
@mock.patch('cursive.certificate_utils.verify_certificate')
def test_get_public_key_invalid_key(self, mock_verify_certificate, mock_get_certificate):
    bad_pub_key = 'A' * 256
    mock_get_certificate.return_value = FakeCryptoCertificate(bad_pub_key)
    sig_key_type = signature_utils.SignatureKeyType.lookup(signature_utils.RSA_PSS)
    self.assertRaisesRegex(exception.SignatureVerificationError, 'Invalid public key type for signature key type: .*', signature_utils.get_public_key, None, None, sig_key_type)