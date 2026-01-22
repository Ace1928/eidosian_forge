from unittest import mock
from heat.engine.resources.openstack.neutron.taas import tap_flow
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_resource_handle_delete_resource_id_is_none(self):
    self.test_resource.resource_id = None
    self.assertIsNone(self.test_resource.handle_delete())
    self.assertEqual(0, self.test_client_plugin.delete_ext_resource.call_count)