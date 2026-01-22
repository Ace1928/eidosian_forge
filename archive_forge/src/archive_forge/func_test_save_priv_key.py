import copy
from unittest import mock
from heat.common import exception
from heat.engine.clients.os import keystone
from heat.engine.clients.os import nova
from heat.engine import resource
from heat.engine.resources.openstack.nova import keypair
from heat.engine import scheduler
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_save_priv_key(self):
    """Test a saved private key."""
    key_name = 'save_private'
    tp_test, created_key = self._get_mock_kp_for_create(key_name, priv_saved=True)
    self.patchobject(self.fake_keypairs, 'get', return_value=created_key)
    scheduler.TaskRunner(tp_test.create)()
    self.assertEqual('private key for save_private', tp_test.FnGetAtt('private_key'))
    self.assertEqual('generated test public key', tp_test.FnGetAtt('public_key'))
    self.assertEqual((tp_test.CREATE, tp_test.COMPLETE), tp_test.state)
    self.assertEqual(tp_test.resource_id, created_key.name)