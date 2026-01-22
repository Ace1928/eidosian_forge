import uuid
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneclient.tests.unit.v2_0 import utils
from keystoneclient.v2_0 import client
from keystoneclient.v2_0 import tenants
from keystoneclient.v2_0 import users
def test_list_marker(self):
    self.stub_url('GET', ['tenants'], json=self.TEST_TENANTS)
    tenant_list = self.client.tenants.list(marker=1)
    self.assertQueryStringIs('marker=1')
    [self.assertIsInstance(t, tenants.Tenant) for t in tenant_list]