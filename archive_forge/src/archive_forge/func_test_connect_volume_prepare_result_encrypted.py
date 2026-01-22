import builtins
import functools
import io
import time
from unittest import mock
import ddt
from os_brick import exception
from os_brick.tests import base
from os_brick import utils
@ddt.data('/dev/md/alias', b'/dev/md/alias')
@mock.patch('os_brick.utils._symlink_name_from_device_path')
@mock.patch('os.path.realpath')
@mock.patch('os_brick.privileged.rootwrap.link_root')
def test_connect_volume_prepare_result_encrypted(self, connector_path, mock_link, mock_path, mock_get_symlink):
    """Test decorator for encrypted device."""
    real_device = '/dev/md-6'
    expected_symlink = '/dev/disk/by-id/os-brick_dev_md_alias'
    mock_path.return_value = real_device
    mock_get_symlink.return_value = expected_symlink
    testing_self = mock.Mock()
    testing_self.connect_volume.return_value = {'type': 'block', 'path': connector_path}
    conn_props = {'encrypted': True}
    func = utils.connect_volume_prepare_result(testing_self.connect_volume)
    res = func(testing_self, conn_props)
    self.assertEqual({'type': 'block', 'path': expected_symlink}, res)
    testing_self.connect_volume.assert_called_once_with(testing_self, conn_props)
    expected_connector_path = utils.convert_str(connector_path)
    mock_get_symlink.assert_called_once_with(expected_connector_path)
    mock_link.assert_called_once_with(real_device, expected_symlink, force=True)