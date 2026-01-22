import ctypes
import struct
from eventlet import patcher
from oslo_log import log as logging
from oslo_utils import units
import six
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi import wintypes
def create_pipe(self, security_attributes=None, size=0, inherit_handle=False):
    """Create an anonymous pipe.

        The main advantage of this method over os.pipe is that it allows
        creating inheritable pipe handles (which is flawed on most Python
        versions).
        """
    r = wintypes.HANDLE()
    w = wintypes.HANDLE()
    if inherit_handle and (not security_attributes):
        security_attributes = wintypes.SECURITY_ATTRIBUTES()
        security_attributes.bInheritHandle = inherit_handle
        security_attributes.nLength = ctypes.sizeof(security_attributes)
    self._run_and_check_output(kernel32.CreatePipe, ctypes.byref(r), ctypes.byref(w), ctypes.byref(security_attributes) if security_attributes else None, size)
    return (r.value, w.value)