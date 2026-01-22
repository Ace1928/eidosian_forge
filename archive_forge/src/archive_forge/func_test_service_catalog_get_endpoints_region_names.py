import uuid
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1.tests.unit import utils
def test_service_catalog_get_endpoints_region_names(self):
    auth_ref = access.create(body=self.AUTH_RESPONSE_BODY)
    sc = auth_ref.service_catalog
    endpoints = sc.get_endpoints(service_type='image', region_name='North')
    self.assertEqual(len(endpoints), 1)
    self.assertEqual(endpoints['image'][0]['publicURL'], 'https://image.north.host/v1/')
    endpoints = sc.get_endpoints(service_type='image', region_name='South')
    self.assertEqual(len(endpoints), 1)
    self.assertEqual(endpoints['image'][0]['publicURL'], 'https://image.south.host/v1/')
    endpoints = sc.get_endpoints(service_type='compute')
    self.assertEqual(len(endpoints['compute']), 2)
    endpoints = sc.get_endpoints(service_type='compute', region_name='North')
    self.assertEqual(len(endpoints['compute']), 2)
    endpoints = sc.get_endpoints(service_type='compute', region_name='West')
    self.assertEqual(len(endpoints['compute']), 0)