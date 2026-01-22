from unittest import mock
import ddt
from os_brick import exception
from os_brick.initiator.connectors import rbd
from os_brick.initiator import linuxrbd
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests.initiator.connectors import test_base_rbd
from os_brick.tests.initiator import test_connector
from os_brick import utils
@mock.patch('os_brick.initiator.connectors.rbd.tempfile.mkstemp')
def test_create_ceph_conf(self, mock_mkstemp):
    mockopen = mock.mock_open()
    fd = mock.sentinel.fd
    tmpfile = mock.sentinel.tmpfile
    mock_mkstemp.return_value = (fd, tmpfile)
    with mock.patch('os.fdopen', mockopen, create=True):
        rbd_connector = rbd.RBDConnector(None)
        conf_path = rbd_connector._create_ceph_conf(self.hosts, self.ports, self.clustername, self.user, self.keyring)
    self.assertEqual(conf_path, tmpfile)
    mock_mkstemp.assert_called_once_with(prefix='brickrbd_')
    _, args, _ = mockopen().writelines.mock_calls[0]
    self.assertIn('[global]', args[0])