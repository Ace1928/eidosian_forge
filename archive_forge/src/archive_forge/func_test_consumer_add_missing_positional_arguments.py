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
@utils.parameterized_dataset({'no_args': [[{}]], 'one_arg_1': [[{'service': 'service1'}]], 'one_arg_2': [[{'resource_type': 'type1'}]], 'one_arg_3': [[{'resource_id': 'id1'}]], 'two_args_1': [[{'service': 'service1', 'resource_type': 'type1'}]], 'two_args_2': [[{'service': 'service1', 'resource_id': 'id1'}]], 'two_args_3': [[{'resource_type': 'type1', 'resource_id': 'id'}]]})
def test_consumer_add_missing_positional_arguments(self, consumers):
    """Missing Positional Arguments - Addition

        Tries to add a secret consumer without providing all of the required
        positional arguments (service, resource_type, resource_id).
        """
    key = test_key_manager._get_test_passphrase()
    self.assertIsNotNone(key)
    stored_id = self.key_mgr.store(self.ctxt, key)
    self.addCleanup(self.key_mgr.delete, self.ctxt, stored_id, True)
    self.assertIsNotNone(stored_id)
    for consumer in consumers:
        e = self.assertRaises(TypeError, self.key_mgr.add_consumer, self.ctxt, stored_id, consumer)
    self.assertIn('register_consumer() missing', str(e))