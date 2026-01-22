from keystoneauth1 import fixture
from keystoneclient import access
from keystoneclient import exceptions
from keystoneclient.tests.unit.v2_0 import client_fixtures
from keystoneclient.tests.unit.v2_0 import utils
def test_service_catalog_param_overrides_body_region(self):
    self.AUTH_RESPONSE_BODY['access']['region_name'] = 'North'
    with self.deprecations.expect_deprecations_here():
        auth_ref = access.AccessInfo.factory(None, self.AUTH_RESPONSE_BODY)
    sc = auth_ref.service_catalog
    url = sc.url_for(service_type='image')
    self.assertEqual(url, 'https://image.north.host/v1/')
    url = sc.url_for(service_type='image', region_name='South')
    self.assertEqual(url, 'https://image.south.host/v1/')
    endpoints = sc.get_endpoints(service_type='image')
    self.assertEqual(len(endpoints['image']), 1)
    self.assertEqual(endpoints['image'][0]['publicURL'], 'https://image.north.host/v1/')
    endpoints = sc.get_endpoints(service_type='image', region_name='South')
    self.assertEqual(len(endpoints['image']), 1)
    self.assertEqual(endpoints['image'][0]['publicURL'], 'https://image.south.host/v1/')