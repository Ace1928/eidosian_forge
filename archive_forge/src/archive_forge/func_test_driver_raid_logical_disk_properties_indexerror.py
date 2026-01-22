from unittest import mock
import testtools
from testtools import matchers
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import driver
@mock.patch.object(driver.DriverManager, '_list', autospec=True)
def test_driver_raid_logical_disk_properties_indexerror(self, _list_mock):
    _list_mock.side_effect = IndexError
    properties = self.mgr.raid_logical_disk_properties(DRIVER2['name'])
    _list_mock.assert_called_once_with(self.mgr, '/v1/drivers/%s/raid/logical_disk_properties' % DRIVER2['name'], os_ironic_api_version=None, global_request_id=None)
    self.assertEqual({}, properties)