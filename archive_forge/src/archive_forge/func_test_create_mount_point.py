from unittest import mock
import ddt
from os_brick import exception
from os_brick.remotefs import windows_remotefs
from os_brick.tests import base
@ddt.data({'use_local_path': True}, {'path_exists': True, 'is_symlink': True}, {'path_exists': True})
@mock.patch.object(windows_remotefs.WindowsRemoteFsClient, 'get_local_share_path')
@mock.patch.object(windows_remotefs.WindowsRemoteFsClient, 'get_mount_point')
@mock.patch.object(windows_remotefs, 'os')
@ddt.unpack
def test_create_mount_point(self, mock_os, mock_get_mount_point, mock_get_local_share_path, path_exists=False, is_symlink=False, use_local_path=False):
    mock_os.path.exists.return_value = path_exists
    mock_os.isdir.return_value = False
    self._pathutils.is_symlink.return_value = is_symlink
    if path_exists and (not is_symlink):
        self.assertRaises(exception.BrickException, self._remotefs._create_mount_point, self._FAKE_SHARE, use_local_path)
    else:
        self._remotefs._create_mount_point(self._FAKE_SHARE, use_local_path)
    mock_get_mount_point.assert_called_once_with(self._FAKE_SHARE)
    mock_os.path.isdir.assert_called_once_with(mock.sentinel.mount_base)
    if use_local_path:
        mock_get_local_share_path.assert_called_once_with(self._FAKE_SHARE)
        expected_symlink_target = mock_get_local_share_path.return_value
    else:
        expected_symlink_target = self._FAKE_SHARE.replace('/', '\\')
    if path_exists:
        self._pathutils.is_symlink.assert_called_once_with(mock_get_mount_point.return_value)
    else:
        self._pathutils.create_sym_link.assert_called_once_with(mock_get_mount_point.return_value, expected_symlink_target)