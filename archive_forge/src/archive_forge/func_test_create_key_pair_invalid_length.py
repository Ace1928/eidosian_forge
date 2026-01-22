from cryptography.hazmat import backends
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from oslo_context import context
from castellan.common import exception
from castellan.common.objects import symmetric_key as sym_key
from castellan.tests.unit.key_manager import mock_key_manager as mock_key_mgr
from castellan.tests.unit.key_manager import test_key_manager as test_key_mgr
def test_create_key_pair_invalid_length(self):
    self.assertRaises(ValueError, self.key_mgr.create_key_pair, self.context, 'RSA', 10)