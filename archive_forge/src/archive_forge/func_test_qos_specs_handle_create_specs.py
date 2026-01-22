from unittest import mock
from heat.engine.clients.os import cinder as c_plugin
from heat.engine.resources.openstack.cinder import qos_specs
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_qos_specs_handle_create_specs(self):
    self._set_up_qos_specs_environment()
    self.assertEqual(1, self.qos_specs.create.call_count)
    self.assertEqual(self.value.id, self.my_qos_specs.resource_id)