from unittest import mock
import ddt
from oslo_utils import units
from oslo_vmware.objects import datastore
from oslo_vmware import vim_util
from os_brick import exception
from os_brick.initiator.connectors import vmware
from os_brick.tests.initiator import test_connector
@mock.patch('os.path.exists')
def test_disconnect_volume_with_missing_temp_file(self, path_exists):
    path_exists.return_value = False
    path = mock.sentinel.path
    self.assertRaises(exception.NotFound, self._connector.disconnect_volume, mock.ANY, {'path': path})
    path_exists.assert_called_once_with(path)