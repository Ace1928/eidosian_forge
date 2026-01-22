import hashlib
import hmac
from oslotest import base as test_base
import testscenarios
from oslo_utils import secretutils
def test_string_data_raises_type_error(self):
    self.assertRaises(TypeError, hashlib.md5, 'foo')
    self.assertRaises(TypeError, secretutils.md5, 'foo')
    self.assertRaises(TypeError, secretutils.md5, 'foo', usedforsecurity=True)
    self.assertRaises(TypeError, secretutils.md5, 'foo', usedforsecurity=False)