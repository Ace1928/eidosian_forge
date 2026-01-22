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
@mock.patch('cursive.signature_utils.get_public_key')
def test_verify_signature_bad_signature(self, mock_get_pub_key):
    data = b'224626ae19824466f2a7f39ab7b80f7f'
    mock_get_pub_key.return_value = TEST_RSA_PRIVATE_KEY.public_key()
    img_sig_cert_uuid = 'fea14bc2-d75f-4ba5-bccc-b5c924ad0693'
    verifier = signature_utils.get_verifier(None, img_sig_cert_uuid, 'SHA-256', 'BLAH', signature_utils.RSA_PSS)
    verifier.update(data)
    self.assertRaises(crypto_exceptions.InvalidSignature, verifier.verify)