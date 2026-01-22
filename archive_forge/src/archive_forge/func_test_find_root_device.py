from unittest import mock
import ddt
from os_brick import exception
from os_brick.initiator.connectors import rbd
from os_brick.initiator import linuxrbd
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests.initiator.connectors import test_base_rbd
from os_brick.tests.initiator import test_connector
from os_brick import utils
@mock.patch.object(rbd.RBDConnector, '_get_rbd_args')
@mock.patch.object(rbd.RBDConnector, '_execute')
def test_find_root_device(self, mock_execute, mock_args):
    mock_args.return_value = [mock.sentinel.rbd_args]
    mock_execute.return_value = ('{"0":{"pool":"pool","device":"/dev/rdb0","name":"image"},"1":{"pool":"pool","device":"/dev/rbd1","name":"fake_volume"}}', 'stderr')
    connector = rbd.RBDConnector(None)
    res = connector._find_root_device(self.connection_properties, mock.sentinel.conf)
    mock_args.assert_called_once_with(self.connection_properties, mock.sentinel.conf)
    mock_execute.assert_called_once_with('rbd', 'showmapped', '--format=json', mock.sentinel.rbd_args, root_helper=connector._root_helper, run_as_root=True)
    self.assertEqual('/dev/rbd1', res)