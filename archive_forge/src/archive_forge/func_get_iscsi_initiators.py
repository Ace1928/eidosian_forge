import ctypes
import functools
import inspect
import socket
import time
from oslo_log import log as logging
import six
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils.storage import diskutils
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi.errmsg import iscsierr
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi.libs import iscsidsc as iscsi_struct
@ensure_buff_and_retrieve_items(struct_type=ctypes.c_wchar, func_requests_buff_sz=False, parse_output=False)
def get_iscsi_initiators(self, buff=None, buff_size=None, element_count=None):
    """Get the list of available iSCSI initiator HBAs."""
    self._run_and_check_output(iscsidsc.ReportIScsiInitiatorListW, ctypes.byref(element_count), buff)
    return self._parse_string_list(buff, element_count.value)