from unittest import mock
import ddt
from os_brick import exception
from os_brick.initiator.connectors import rbd
from os_brick.initiator import linuxrbd
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests.initiator.connectors import test_base_rbd
from os_brick.tests.initiator import test_connector
from os_brick import utils
@ddt.data('\n        [{"id":"0","pool":"pool","device":"/dev/rbd0","name":"image"},\n         {"id":"1","pool":"pool","device":"/dev/rdb1","name":"image_2"}]\n        ', '\n        {"0":{"pool":"pool","device":"/dev/rbd0","name":"image"},\n         "1":{"pool":"pool","device":"/dev/rdb1","name":"image_2"}}\n        ')
@mock.patch('os_brick.privileged.rbd.delete_if_exists')
@mock.patch.object(priv_rootwrap, 'execute', return_value=None)
def test_disconnect_local_volume(self, rbd_map_out, mock_execute, mock_delete):
    """Test the disconnect volume case with local attach."""
    rbd_connector = rbd.RBDConnector(None, do_local_attach=True)
    conn = {'name': 'pool/image', 'auth_username': 'fake_user', 'hosts': ['192.168.10.2'], 'ports': ['6789']}
    mock_execute.side_effect = [(rbd_map_out, None), (None, None)]
    show_cmd = ['rbd', 'showmapped', '--format=json', '--id', 'fake_user', '--mon_host', '192.168.10.2:6789']
    unmap_cmd = ['rbd', 'unmap', '/dev/rbd0', '--id', 'fake_user', '--mon_host', '192.168.10.2:6789']
    rbd_connector.disconnect_volume(conn, None)
    mock_execute.assert_has_calls([mock.call(*show_cmd, root_helper=None, run_as_root=True), mock.call(*unmap_cmd, root_helper=None, run_as_root=True)])
    mock_delete.assert_not_called()