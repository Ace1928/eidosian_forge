import base64
import hashlib
import hmac
from unittest import mock
import uuid
from osprofiler import _utils as utils
from osprofiler.tests import test
def test_signed_unpack_no_key_or_hmac_data(self):
    data = {'some': 'data'}
    packed_data, hmac_data = utils.signed_pack(data, 'secret')
    self.assertIsNone(utils.signed_unpack(packed_data, hmac_data, None))
    self.assertIsNone(utils.signed_unpack(packed_data, None, 'secret'))
    self.assertIsNone(utils.signed_unpack(packed_data, ' ', 'secret'))