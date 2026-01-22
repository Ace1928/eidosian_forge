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
@mock.patch.object(clusterutils.ClusterUtils, '_get_cluster_nodes')
def test_check_cluster_state_not_enough_nodes(self, mock_get_nodes):
    self.assertRaises(exceptions.HyperVClusterException, self._clusterutils.check_cluster_state)