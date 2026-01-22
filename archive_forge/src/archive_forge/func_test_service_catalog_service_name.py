import uuid
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1.tests.unit import utils
def test_service_catalog_service_name(self):
    auth_ref = access.create(body=self.AUTH_RESPONSE_BODY)
    sc = auth_ref.service_catalog
    url = sc.url_for(service_name='Image Servers', interface='public', service_type='image', region_name='North')
    self.assertEqual('https://image.north.host/v1/', url)
    self.assertRaises(exceptions.EndpointNotFound, sc.url_for, service_name='Image Servers', service_type='compute')
    urls = sc.get_urls(service_type='image', service_name='Image Servers', interface='public')
    self.assertIn('https://image.north.host/v1/', urls)
    self.assertIn('https://image.south.host/v1/', urls)
    urls = sc.get_urls(service_type='image', service_name='Servers', interface='public')
    self.assertEqual(0, len(urls))