from unittest import mock
import ddt
from oslo_concurrency import processutils as putils
from os_brick import exception
from os_brick import privileged
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests import base
@mock.patch.object(priv_rootwrap.unlink_root.privsep_entrypoint, 'client_mode', False)
@mock.patch('os.symlink')
@mock.patch('os.remove', side_effect=IndexError)
def test_link_root_force_fail(self, mock_remove, mock_link):
    self.assertRaises(IndexError, priv_rootwrap.link_root, mock.sentinel.target, mock.sentinel.link_name)
    mock_remove.assert_called_once_with(mock.sentinel.link_name)
    mock_link.assert_not_called()