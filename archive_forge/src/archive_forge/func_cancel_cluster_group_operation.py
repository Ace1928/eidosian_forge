import contextlib
import ctypes
from os_win._i18n import _
from os_win import constants
from os_win import exceptions
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi.libs import clusapi as clusapi_def
from os_win.utils.winapi import wintypes
def cancel_cluster_group_operation(self, group_handle):
    """Requests a pending move operation to be canceled.

        This only applies to move operations requested by
        MoveClusterGroup(Ex), thus it will not apply to fail overs.

        return: True if the cancel request completed successfuly,
                False if it's still in progress.
        """
    ret_val = self._run_and_check_output(clusapi.CancelClusterGroupOperation, group_handle, 0, ignored_error_codes=[w_const.ERROR_IO_PENDING])
    cancel_completed = ret_val != w_const.ERROR_IO_PENDING
    return cancel_completed