import base64
import hashlib
import hmac
from unittest import mock
import uuid
from osprofiler import _utils as utils
from osprofiler.tests import test
def test_signed_pack_unpack_many_keys(self):
    keys = ['secret', 'secret2', 'secret3']
    data = {'some': 'data'}
    packed_data, hmac_data = utils.signed_pack(data, keys[-1])
    process_data = utils.signed_unpack(packed_data, hmac_data, keys)
    self.assertEqual(keys[-1], process_data['hmac_key'])