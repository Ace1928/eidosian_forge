import testtools
from unittest import mock
from troveclient.compat import auth
from troveclient.compat import exceptions
def test__load(self):
    url = 'random_url'
    auth.ServiceCatalog._url_for = mock.Mock(return_value=url)
    scObj = auth.ServiceCatalog()
    scObj.region = None
    scObj.service_url = None
    scObj._load()
    self.assertEqual(url, scObj.public_url)
    self.assertEqual(url, scObj.management_url)
    service_url = 'service_url'
    scObj = auth.ServiceCatalog()
    scObj.region = None
    scObj.service_url = service_url
    scObj._load()
    self.assertEqual(service_url, scObj.public_url)
    self.assertEqual(service_url, scObj.management_url)