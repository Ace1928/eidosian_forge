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
def cancel_cluster_group_migration(self, group_name, expected_state, timeout=None):
    with self._cmgr.open_cluster() as cluster_handle, self._cmgr.open_cluster_group(group_name, cluster_handle=cluster_handle) as group_handle, _ClusterGroupStateChangeListener(cluster_handle, group_name) as listener:
        self._cancel_cluster_group_migration(listener, group_name, group_handle, expected_state, timeout)