import base64
import hashlib
import hmac
from unittest import mock
import uuid
from osprofiler import _utils as utils
from osprofiler.tests import test
def test_generate_hmac(self):
    hmac_key = 'secrete'
    data = 'my data'
    h = hmac.new(utils.binary_encode(hmac_key), digestmod=hashlib.sha1)
    h.update(utils.binary_encode(data))
    self.assertEqual(h.hexdigest(), utils.generate_hmac(data, hmac_key))