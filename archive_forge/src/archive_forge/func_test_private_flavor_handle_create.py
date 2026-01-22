from unittest import mock
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_private_flavor_handle_create(self):
    self.create_flavor(is_public=False)
    value = mock.MagicMock()
    flavor_id = '927202df-1afb-497f-8368-9c2d2f26e5db'
    value.id = flavor_id
    value.is_public = False
    self.flavors.create.return_value = value
    self.flavors.get.return_value = value
    self.my_flavor.handle_create()
    value.set_keys.assert_called_once_with({'foo': 'bar'})
    self.assertEqual(flavor_id, self.my_flavor.resource_id)
    self.assertFalse(self.my_flavor.FnGetAtt('is_public'))
    client_test = self.my_flavor.client().flavor_access.add_tenant_access
    test_tenants = [mock.call(value, 'foo'), mock.call(value, 'bar')]
    self.assertEqual(test_tenants, client_test.call_args_list)