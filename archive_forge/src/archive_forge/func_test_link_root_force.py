from unittest import mock
import ddt
from oslo_concurrency import processutils as putils
from os_brick import exception
from os_brick import privileged
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests import base
@ddt.data(None, FileNotFoundError)
@mock.patch.object(priv_rootwrap.unlink_root.privsep_entrypoint, 'client_mode', False)
@mock.patch('os.symlink')
@mock.patch('os.remove')
def test_link_root_force(self, remove_effect, mock_remove, mock_link):
    mock_remove.side_effect = remove_effect
    priv_rootwrap.link_root(mock.sentinel.target, mock.sentinel.link_name)
    mock_remove.assert_called_once_with(mock.sentinel.link_name)
    mock_link.assert_called_once_with(mock.sentinel.target, mock.sentinel.link_name)