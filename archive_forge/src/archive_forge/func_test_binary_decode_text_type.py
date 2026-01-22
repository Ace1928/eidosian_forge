import base64
import hashlib
import hmac
from unittest import mock
import uuid
from osprofiler import _utils as utils
from osprofiler.tests import test
def test_binary_decode_text_type(self):
    self.assertEqual('text', utils.binary_decode('text'))