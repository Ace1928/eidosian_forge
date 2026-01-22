import abc
from keystoneauth1 import identity
from keystoneauth1 import session
from oslo_config import cfg
from oslo_context import context
from oslo_utils import uuidutils
from oslotest import base
from testtools import testcase
from castellan.common.credentials import keystone_password
from castellan.common.credentials import keystone_token
from castellan.common import exception
from castellan.key_manager import barbican_key_manager
from castellan.tests.functional import config
from castellan.tests.functional.key_manager import test_key_manager
from castellan.tests import utils
def test_secret_create_check_empty_consumers_list(self):
    """Check that the consumers entity is a list and is empty."""
    key = test_key_manager._get_test_passphrase()
    self.assertIsNotNone(key)
    stored_id = self.key_mgr.store(self.ctxt, key)
    self.addCleanup(self.key_mgr.delete, self.ctxt, stored_id, True)
    self.assertIsNotNone(stored_id)
    resp = self.key_mgr.get(self.ctxt, stored_id)
    consumers = resp.consumers
    self.assertIsInstance(consumers, list)
    self.assertEqual(len(consumers), 0)