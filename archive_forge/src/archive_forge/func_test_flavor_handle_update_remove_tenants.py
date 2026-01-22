from unittest import mock
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_flavor_handle_update_remove_tenants(self):
    self.create_flavor(is_public=False)
    value = mock.MagicMock()
    new_tenants = []
    prop_diff = {'tenants': new_tenants}
    self.flavors.get.return_value = value
    itemFoo = mock.MagicMock()
    itemFoo.tenant_id = 'foo'
    itemBar = mock.MagicMock()
    itemBar.tenant_id = 'bar'
    self.my_flavor.client().flavor_access.list.return_value = [itemFoo, itemBar]
    self.my_flavor.handle_update(json_snippet=None, tmpl_diff=None, prop_diff=prop_diff)
    test_tenants_remove = [mock.call(value, 'foo'), mock.call(value, 'bar')]
    test_rem = self.my_flavor.client().flavor_access.remove_tenant_access
    self.assertCountEqual(test_tenants_remove, test_rem.call_args_list)