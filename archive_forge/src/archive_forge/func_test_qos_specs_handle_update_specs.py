from unittest import mock
from heat.engine.clients.os import cinder as c_plugin
from heat.engine.resources.openstack.cinder import qos_specs
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_qos_specs_handle_update_specs(self):
    self._set_up_qos_specs_environment()
    resource_id = self.my_qos_specs.resource_id
    prop_diff = {'specs': {'foo': 'bar', 'bar': 'bar'}}
    set_expected = {'bar': 'bar'}
    unset_expected = ['foo1']
    self.my_qos_specs.handle_update(json_snippet=None, tmpl_diff=None, prop_diff=prop_diff)
    self.qos_specs.set_keys.assert_called_once_with(resource_id, set_expected)
    self.qos_specs.unset_keys.assert_called_once_with(resource_id, unset_expected)