import uuid
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1.tests.unit import utils
def test_service_catalog_get_url_region_names(self):
    sc = access.create(auth_token=uuid.uuid4().hex, body=self.AUTH_RESPONSE_BODY).service_catalog
    urls = sc.get_urls(service_type='image')
    self.assertEqual(len(urls), 2)
    urls = sc.get_urls(service_type='image', region_name='North')
    self.assertEqual(len(urls), 1)
    self.assertEqual(urls[0], self.north_endpoints['public'])
    urls = sc.get_urls(service_type='image', region_name='South')
    self.assertEqual(len(urls), 1)
    self.assertEqual(urls[0], self.south_endpoints['public'])
    urls = sc.get_urls(service_type='image', region_name='West')
    self.assertEqual(len(urls), 0)