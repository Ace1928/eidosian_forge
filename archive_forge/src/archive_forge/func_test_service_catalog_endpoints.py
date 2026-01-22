import uuid
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1.tests.unit import utils
def test_service_catalog_endpoints(self):
    auth_ref = access.create(body=self.AUTH_RESPONSE_BODY)
    sc = auth_ref.service_catalog
    public_ep = sc.get_endpoints(service_type='compute', interface='publicURL')
    self.assertEqual(public_ep['compute'][1]['tenantId'], '2')
    self.assertEqual(public_ep['compute'][1]['versionId'], '1.1')
    self.assertEqual(public_ep['compute'][1]['internalURL'], 'https://compute.north.host/v1.1/3456')