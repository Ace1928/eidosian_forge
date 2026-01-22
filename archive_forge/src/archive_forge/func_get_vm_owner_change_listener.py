import ctypes
import re
import sys
import threading
import time
from eventlet import patcher
from eventlet import tpool
from oslo_log import log as logging
from oslo_utils import excutils
from six.moves import queue
from os_win._i18n import _
from os_win import _utils
import os_win.conf
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils.compute import _clusapi_utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi.libs import clusapi as clusapi_def
from os_win.utils.winapi import wintypes
def get_vm_owner_change_listener(self):

    def listener(callback):
        watcher = self._get_failover_watcher()
        while True:
            try:
                self._monitor_vm_failover(watcher, callback, constants.DEFAULT_WMI_EVENT_TIMEOUT_MS)
            except Exception:
                LOG.exception('The VM cluster group owner change event listener encountered an unexpected exception.')
                time.sleep(constants.DEFAULT_WMI_EVENT_TIMEOUT_MS / 1000)
    return listener