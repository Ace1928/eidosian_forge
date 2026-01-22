import binascii
from unittest import mock
from oslo_config import cfg
from castellan.common import exception
from castellan.common.objects import symmetric_key as key
from castellan import key_manager
from castellan.key_manager import not_implemented_key_manager
from castellan.tests.unit.key_manager import test_key_manager
def test_delete_fixed_key(self):
    self.key_mgr.delete(self.ctxt, self.fixed_key_id)
    secret = self.key_mgr.get(self.ctxt, self.fixed_key_id)
    self.assertEqual(self.fixed_key_secret, secret)