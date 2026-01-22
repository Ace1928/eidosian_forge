from unittest import mock
import ddt
from os_brick import exception
from os_brick.initiator.connectors import rbd
from os_brick.initiator import linuxrbd
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests.initiator.connectors import test_base_rbd
from os_brick.tests.initiator import test_connector
from os_brick import utils
@mock.patch('os_brick.initiator.linuxrbd.rbd')
@mock.patch('os_brick.initiator.linuxrbd.rados')
@mock.patch.object(rbd.RBDConnector, '_create_ceph_conf')
@mock.patch('os.path.exists')
def test_provided_keyring(self, mock_path, mock_conf, mock_rados, mock_rbd):
    conn = rbd.RBDConnector(None)
    mock_path.return_value = False
    mock_conf.return_value = '/tmp/fake_dir/fake_ceph.conf'
    self.connection_properties['keyring'] = self.keyring
    conn.connect_volume(self.connection_properties)
    mock_conf.assert_called_once_with(self.hosts, self.ports, self.clustername, self.user, self.keyring)