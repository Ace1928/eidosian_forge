import ctypes
from unittest import mock
import ddt
from six.moves import queue
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.compute import clusterutils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi.libs import clusapi as clusapi_def
from os_win.utils.winapi import wintypes
@mock.patch.object(clusterutils.ClusterUtils, '_get_cluster_group_state')
@mock.patch.object(clusterutils.ClusterUtils, '_is_migration_pending')
@mock.patch.object(clusterutils.ClusterUtils, '_wait_for_cluster_group_migration')
@ddt.data({}, {'cancel_exception': test_base.TestingException()}, {'cancel_exception': exceptions.Win32Exception(error_code=w_const.INVALID_HANDLE_VALUE, func_name=mock.sentinel.func_name, error_message=mock.sentinel.error_message)}, {'cancel_exception': exceptions.Win32Exception(error_code=w_const.ERROR_INVALID_STATE, func_name=mock.sentinel.func_name, error_message=mock.sentinel.error_message), 'invalid_state_for_cancel': True}, {'cancel_exception': exceptions.Win32Exception(error_code=w_const.ERROR_INVALID_STATE, func_name=mock.sentinel.func_name, error_message=mock.sentinel.error_message), 'invalid_state_for_cancel': True, 'cancel_still_pending': True}, {'cancel_still_pending': True}, {'cancel_still_pending': True, 'cancel_wait_exception': test_base.TestingException()})
@ddt.unpack
def test_cancel_cluster_group_migration(self, mock_wait_migr, mock_is_migr_pending, mock_get_gr_state, cancel_still_pending=False, cancel_exception=None, invalid_state_for_cancel=False, cancel_wait_exception=None):
    expected_exception = None
    if cancel_wait_exception:
        expected_exception = exceptions.JobTerminateFailed()
    if cancel_exception and (not invalid_state_for_cancel or cancel_still_pending):
        expected_exception = cancel_exception
    mock_is_migr_pending.return_value = cancel_still_pending
    mock_get_gr_state.return_value = dict(state=mock.sentinel.state, status_info=mock.sentinel.status_info)
    self._clusapi.cancel_cluster_group_operation.side_effect = cancel_exception or (not cancel_still_pending,)
    mock_wait_migr.side_effect = cancel_wait_exception
    cancel_args = (mock.sentinel.listener, mock.sentinel.group_name, mock.sentinel.group_handle, mock.sentinel.expected_state, mock.sentinel.timeout)
    if expected_exception:
        self.assertRaises(expected_exception.__class__, self._clusterutils._cancel_cluster_group_migration, *cancel_args)
    else:
        self._clusterutils._cancel_cluster_group_migration(*cancel_args)
    self._clusapi.cancel_cluster_group_operation.assert_called_once_with(mock.sentinel.group_handle)
    if isinstance(cancel_exception, exceptions.Win32Exception):
        mock_get_gr_state.assert_called_once_with(mock.sentinel.group_handle)
        mock_is_migr_pending.assert_called_once_with(mock.sentinel.state, mock.sentinel.status_info, mock.sentinel.expected_state)
    if cancel_still_pending and (not cancel_exception):
        mock_wait_migr.assert_called_once_with(mock.sentinel.listener, mock.sentinel.group_name, mock.sentinel.group_handle, mock.sentinel.expected_state, timeout=mock.sentinel.timeout)