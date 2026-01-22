import errno
import os
from eventlet import patcher
from oslo_log import log as logging
from os_win._i18n import _
from os_win import constants
from os_win import exceptions
from os_win.utils.io import ioutils
from os_win.utils.winapi import constants as w_const
def flush_log_file(self):
    try:
        self._log_file_handle.flush()
    except (AttributeError, ValueError):
        pass