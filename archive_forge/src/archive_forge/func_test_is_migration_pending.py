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
def test_is_migration_pending(self):
    self.assertTrue(self._clusterutils._is_migration_pending(group_state=constants.CLUSTER_GROUP_OFFLINE, group_status_info=0, expected_state=constants.CLUSTER_GROUP_ONLINE))
    self.assertTrue(self._clusterutils._is_migration_pending(group_state=constants.CLUSTER_GROUP_ONLINE, group_status_info=w_const.CLUSGRP_STATUS_WAITING_IN_QUEUE_FOR_MOVE | 1, expected_state=constants.CLUSTER_GROUP_ONLINE))
    self.assertFalse(self._clusterutils._is_migration_pending(group_state=constants.CLUSTER_GROUP_OFFLINE, group_status_info=0, expected_state=constants.CLUSTER_GROUP_OFFLINE))