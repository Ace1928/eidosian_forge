import uuid
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneclient.tests.unit.v2_0 import utils
from keystoneclient.v2_0 import client
from keystoneclient.v2_0 import tenants
from keystoneclient.v2_0 import users
def test_update_empty_description(self):
    req_body = {'tenant': {'id': self.EXTRAS_ID, 'name': 'tenantX', 'description': '', 'enabled': False}}
    resp_body = {'tenant': {'name': 'tenantX', 'enabled': False, 'id': self.EXTRAS_ID, 'description': ''}}
    self.stub_url('POST', ['tenants', self.EXTRAS_ID], json=resp_body)
    tenant = self.client.tenants.update(req_body['tenant']['id'], req_body['tenant']['name'], req_body['tenant']['description'], req_body['tenant']['enabled'])
    self.assertIsInstance(tenant, tenants.Tenant)
    self.assertRequestBodyIs(json=req_body)
    self.assertEqual(tenant.id, self.EXTRAS_ID)
    self.assertEqual(tenant.name, 'tenantX')
    self.assertEqual(tenant.description, '')
    self.assertFalse(tenant.enabled)