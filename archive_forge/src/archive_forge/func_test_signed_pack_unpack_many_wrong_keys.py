import base64
import hashlib
import hmac
from unittest import mock
import uuid
from osprofiler import _utils as utils
from osprofiler.tests import test
def test_signed_pack_unpack_many_wrong_keys(self):
    keys = ['secret', 'secret2', 'secret3']
    data = {'some': 'data'}
    packed_data, hmac_data = utils.signed_pack(data, 'password')
    process_data = utils.signed_unpack(packed_data, hmac_data, keys)
    self.assertIsNone(process_data)