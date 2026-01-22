import base64
import hashlib
import hmac
from unittest import mock
import uuid
from osprofiler import _utils as utils
from osprofiler.tests import test
def test_signed_unpack_invalid_json(self):
    hmac = 'secret'
    data = base64.urlsafe_b64encode(utils.binary_encode('not_a_json'))
    hmac_data = utils.generate_hmac(data, hmac)
    self.assertIsNone(utils.signed_unpack(data, hmac_data, hmac))