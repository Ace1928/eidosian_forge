import uuid
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1.tests.unit import utils
def test_service_catalog_alias_find_official(self):
    auth_ref = access.create(auth_token=uuid.uuid4().hex, body=self.AUTH_RESPONSE_BODY)
    sc = auth_ref.service_catalog
    public_ep = sc.get_endpoints(service_type='volume', interface='public', region_name='North')
    self.assertEqual(public_ep['block-storage'][0]['region'], 'North')
    self.assertEqual(public_ep['block-storage'][0]['url'], 'http://cinder.north.host/cinderapi/public')