from unittest import mock
import testtools
from testtools import matchers
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import driver
@mock.patch.object(driver.DriverManager, 'delete', autospec=True)
def test_vendor_passthru_unknown_http_method(self, delete_mock):
    kwargs = {'driver_name': 'driver_name', 'method': 'method', 'http_method': 'UNKNOWN'}
    self.assertRaises(exc.InvalidAttribute, self.mgr.vendor_passthru, **kwargs)