import ipaddress
from openstack import exceptions
from openstack.tests.functional import base
def test_update_router_name(self):
    router = self._create_and_verify_advanced_router(external_cidr=u'10.7.7.0/24')
    new_name = self.router_prefix + '_update_name'
    updated = self.operator_cloud.update_router(router['id'], name=new_name)
    self.assertIsNotNone(updated)
    for field in EXPECTED_TOPLEVEL_FIELDS:
        self.assertIn(field, updated)
    self.assertEqual(new_name, updated['name'])
    self.assertEqual(router['status'], updated['status'])
    self.assertEqual(router['admin_state_up'], updated['admin_state_up'])
    self.assertEqual(router['external_gateway_info'], updated['external_gateway_info'])