from unittest import mock
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_flavor_handle_update_add_tenants(self):
    self.create_flavor(is_public=False)
    value = mock.MagicMock()
    new_tenants = ['new_foo', 'new_bar']
    prop_diff = {'tenants': new_tenants}
    self.flavors.get.return_value = value
    self.my_flavor.handle_update(json_snippet=None, tmpl_diff=None, prop_diff=prop_diff)
    test_tenants_add = [mock.call(value, 'new_foo'), mock.call(value, 'new_bar')]
    test_add = self.my_flavor.client().flavor_access.add_tenant_access
    self.assertCountEqual(test_tenants_add, test_add.call_args_list)