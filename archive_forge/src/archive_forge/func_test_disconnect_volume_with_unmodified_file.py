from unittest import mock
import ddt
from oslo_utils import units
from oslo_vmware.objects import datastore
from oslo_vmware import vim_util
from os_brick import exception
from os_brick.initiator.connectors import vmware
from os_brick.tests.initiator import test_connector
@mock.patch('os.path.exists')
@mock.patch('os.path.getmtime')
@mock.patch.object(VMDK_CONNECTOR, '_disconnect')
@mock.patch('os.remove')
def test_disconnect_volume_with_unmodified_file(self, remove, disconnect, getmtime, path_exists):
    path_exists.return_value = True
    mtime = 1467802060
    getmtime.return_value = mtime
    path = mock.sentinel.path
    self._connector.disconnect_volume(mock.ANY, {'path': path, 'last_modified': mtime})
    path_exists.assert_called_once_with(path)
    getmtime.assert_called_once_with(path)
    disconnect.assert_not_called()
    remove.assert_called_once_with(path)