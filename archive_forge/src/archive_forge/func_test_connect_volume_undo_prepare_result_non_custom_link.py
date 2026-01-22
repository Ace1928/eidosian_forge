import builtins
import functools
import io
import time
from unittest import mock
import ddt
from os_brick import exception
from os_brick.tests import base
from os_brick import utils
@ddt.data(({'device_path': '/dev/md/alias'}, {}), ({'device_path': '/dev/md/alias', 'encrypted': False}, None), ({'device_path': '/dev/md/alias'}, {'path': '/dev/md/alias'}), ({'device_path': '/dev/md/alias', 'encrypted': False}, {'path': '/dev/md/alias'}), ({'device_path': io.StringIO(), 'encrypted': True}, None), ({'device_path': '/dev/disk/by-id/wwn-...', 'encrypted': True}, None))
@ddt.unpack
@mock.patch('os_brick.utils._device_path_from_symlink')
@mock.patch('os_brick.privileged.rootwrap.unlink_root')
def test_connect_volume_undo_prepare_result_non_custom_link(outer_self, conn_props, dev_info, mock_unlink, mock_dev_path):

    class Test(object):

        @utils.connect_volume_undo_prepare_result(unlink_after=True)
        def disconnect_volume(self, connection_properties, device_info, force=False, ignore_errors=False):
            outer_self.assertEqual(conn_props, connection_properties)
            outer_self.assertEqual(dev_info, device_info)
            return 'disconnect_volume'

        @utils.connect_volume_undo_prepare_result
        def extend_volume(self, connection_properties):
            outer_self.assertEqual(conn_props, connection_properties)
            return 'extend_volume'
    path = conn_props['device_path']
    mock_dev_path.return_value = path
    t = Test()
    res = t.disconnect_volume(conn_props, dev_info)
    outer_self.assertEqual('disconnect_volume', res)
    res = t.extend_volume(conn_props)
    outer_self.assertEqual('extend_volume', res)
    if conn_props.get('encrypted'):
        outer_self.assertEqual(2, mock_dev_path.call_count)
        mock_dev_path.assert_has_calls((mock.call(path), mock.call(path)))
    else:
        mock_dev_path.assert_not_called()
    mock_unlink.assert_not_called()