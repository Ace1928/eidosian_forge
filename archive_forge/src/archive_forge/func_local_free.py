import ctypes
from oslo_log import log as logging
from os_win import _utils
from os_win import exceptions
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi import wintypes
def local_free(self, handle):
    try:
        self._run_and_check_output(kernel32.LocalFree, handle)
    except exceptions.Win32Exception:
        LOG.exception('Could not deallocate memory. There could be a memory leak.')