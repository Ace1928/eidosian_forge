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
def test_consumer_add_two_remove_one_check_consumers_list(self):
    """Consumers addition and removal - check of list consistency

        Adds two consumers, removes one and verifies if the consumers
        list's length is consistent (equals to 1).
        """
    key = test_key_manager._get_test_passphrase()
    self.assertIsNotNone(key)
    stored_id = self.key_mgr.store(self.ctxt, key)
    self.addCleanup(self.key_mgr.delete, self.ctxt, stored_id, True)
    self.assertIsNotNone(stored_id)
    consumers = [{'service': 'service1', 'resource_type': 'type1', 'resource_id': 'id1'}, {'service': 'service2', 'resource_type': 'type2', 'resource_id': 'id2'}]
    for consumer in consumers:
        self.key_mgr.add_consumer(self.ctxt, stored_id, consumer)
    stored_secret = self.key_mgr.get(self.ctxt, stored_id)
    self.assertCountEqual(consumers, stored_secret.consumers)
    self.key_mgr.remove_consumer(self.ctxt, stored_id, consumers[0])
    stored_secret = self.key_mgr.get(self.ctxt, stored_id)
    self.assertCountEqual(consumers[1:], stored_secret.consumers)