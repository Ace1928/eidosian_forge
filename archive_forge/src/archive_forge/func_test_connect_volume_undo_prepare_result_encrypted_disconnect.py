import builtins
import functools
import io
import time
from unittest import mock
import ddt
from os_brick import exception
from os_brick.tests import base
from os_brick import utils
@mock.patch('os_brick.utils._device_path_from_symlink')
@mock.patch('os_brick.privileged.rootwrap.unlink_root')
def test_connect_volume_undo_prepare_result_encrypted_disconnect(outer_self, mock_unlink, mock_dev_path):
    connector_path = '/dev/md/alias'
    mock_dev_path.return_value = connector_path
    symlink_path = '/dev/disk/by-id/os-brick_dev_md_alias'
    mock_unlink.side_effect = ValueError

    class Test(object):

        @utils.connect_volume_undo_prepare_result(unlink_after=True)
        def disconnect_volume(self, connection_properties, device_info, force=False, ignore_errors=False):
            outer_self.assertEqual(connector_path, connection_properties['device_path'])
            outer_self.assertEqual(connector_path, device_info['path'])
            return 'disconnect_volume'
    conn_props = {'target_portal': '198.72.124.185:3260', 'target_iqn': 'iqn.2010-10.org.openstack:volume-uuid', 'target_lun': 0, 'encrypted': True, 'device_path': symlink_path}
    dev_info = {'type': 'block', 'path': symlink_path}
    t = Test()
    res = t.disconnect_volume(conn_props, dev_info)
    outer_self.assertEqual('disconnect_volume', res)
    mock_dev_path.assert_called_once_with(symlink_path)
    mock_unlink.assert_called_once_with(symlink_path)