from unittest import mock
from heat.engine.clients.os import cinder as c_plugin
from heat.engine.resources.openstack.cinder import qos_specs
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_qos_specs_handle_delete_specs(self):
    self._set_up_qos_specs_environment()
    resource_id = self.my_qos_specs.resource_id
    self.my_qos_specs.handle_delete()
    self.qos_specs.disassociate_all.assert_called_once_with(resource_id)