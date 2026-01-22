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
def test_get_verifier_invalid_image_props(self):
    self.assertRaisesRegex(exception.SignatureVerificationError, 'Required image properties for signature verification do not exist. Cannot verify signature. Missing property: .*', signature_utils.get_verifier, None, None, 'SHA-256', 'BLAH', signature_utils.RSA_PSS)