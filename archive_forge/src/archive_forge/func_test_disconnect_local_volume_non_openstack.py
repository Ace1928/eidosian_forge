from unittest import mock
import ddt
from os_brick import exception
from os_brick.initiator.connectors import rbd
from os_brick.initiator import linuxrbd
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests.initiator.connectors import test_base_rbd
from os_brick.tests.initiator import test_connector
from os_brick import utils
@mock.patch('os_brick.privileged.rbd.delete_if_exists')
@mock.patch.object(rbd.RBDConnector, '_find_root_device')
@mock.patch.object(rbd.RBDConnector, '_execute')
def test_disconnect_local_volume_non_openstack(self, mock_execute, mock_find, mock_delete):
    connector = rbd.RBDConnector(None, do_local_attach=True)
    mock_find.return_value = '/dev/rbd0'
    connector.disconnect_volume(self.connection_properties, {'conf': mock.sentinel.conf})
    mock_find.assert_called_once_with(self.connection_properties, mock.sentinel.conf)
    mock_execute.assert_called_once_with('rbd', 'unmap', '/dev/rbd0', '--id', 'fake_user', '--mon_host', '192.168.10.2:6789', '--conf', mock.sentinel.conf, root_helper=connector._root_helper, run_as_root=True)
    mock_delete.assert_called_once_with(mock.sentinel.conf)