import base64
import hashlib
import hmac
from unittest import mock
import uuid
from osprofiler import _utils as utils
from osprofiler.tests import test
def test_binary_encode_binary_type(self):
    binary = utils.binary_encode('text')
    self.assertEqual(binary, utils.binary_encode(binary))