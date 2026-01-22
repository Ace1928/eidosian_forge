import ipaddress
from openstack import exceptions
from openstack.tests.functional import base
def test_update_router_admin_state(self):
    router = self._create_and_verify_advanced_router(external_cidr=u'10.8.8.0/24')
    updated = self.operator_cloud.update_router(router['id'], admin_state_up=True)
    self.assertIsNotNone(updated)
    for field in EXPECTED_TOPLEVEL_FIELDS:
        self.assertIn(field, updated)
    self.assertTrue(updated['admin_state_up'])
    self.assertNotEqual(router['admin_state_up'], updated['admin_state_up'])
    self.assertEqual(router['status'], updated['status'])
    self.assertEqual(router['name'], updated['name'])
    self.assertEqual(router['external_gateway_info'], updated['external_gateway_info'])