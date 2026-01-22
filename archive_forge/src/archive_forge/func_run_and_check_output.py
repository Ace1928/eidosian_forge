import ctypes
from oslo_log import log as logging
from os_win import _utils
from os_win import exceptions
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi import wintypes
def run_and_check_output(self, *args, **kwargs):
    eventlet_nonblocking_mode = kwargs.pop('eventlet_nonblocking_mode', True)
    if eventlet_nonblocking_mode:
        return _utils.avoid_blocking_call(self._run_and_check_output, *args, **kwargs)
    else:
        return self._run_and_check_output(*args, **kwargs)