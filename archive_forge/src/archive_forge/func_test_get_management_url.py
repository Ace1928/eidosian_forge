import testtools
from unittest import mock
from troveclient.compat import auth
from troveclient.compat import exceptions
def test_get_management_url(self):
    test_mng_url = 'test_management_url'
    scObj = auth.ServiceCatalog()
    scObj.management_url = test_mng_url
    self.assertEqual(test_mng_url, scObj.get_management_url())