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
def test_get_public_key_dsa(self, mock_verify_cert, mock_get_cert):
    fake_cert = FakeCryptoCertificate(TEST_DSA_PRIVATE_KEY.public_key())
    mock_get_cert.return_value = fake_cert
    sig_key_type = signature_utils.SignatureKeyType.lookup(signature_utils.DSA)
    result_pub_key = signature_utils.get_public_key(None, None, sig_key_type)
    self.assertEqual(fake_cert.public_key(), result_pub_key)