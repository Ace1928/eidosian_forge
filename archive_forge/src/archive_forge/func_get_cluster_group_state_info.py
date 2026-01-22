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
def get_cluster_group_state_info(self, group_name):
    """Gets cluster group state info.

        :return: a dict containing the following keys:
            ['state', 'migration_queued', 'owner_node']
        """
    with self._cmgr.open_cluster_group(group_name) as group_handle:
        state_info = self._get_cluster_group_state(group_handle)
        migration_queued = self._is_migration_queued(state_info['status_info'])
        return dict(owner_node=state_info['owner_node'], state=state_info['state'], migration_queued=migration_queued)