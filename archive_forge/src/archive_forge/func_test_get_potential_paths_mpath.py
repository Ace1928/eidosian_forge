import collections
import os
from unittest import mock
import ddt
from oslo_concurrency import processutils as putils
from os_brick import exception
from os_brick.initiator.connectors import iscsi
from os_brick.initiator import linuxscsi
from os_brick.initiator import utils
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests.initiator import test_connector
@mock.patch.object(iscsi.ISCSIConnector, '_get_device_path')
def test_get_potential_paths_mpath(self, get_path_mock):
    self.connector.use_multipath = True
    res = self.connector._get_potential_volume_paths(self.CON_PROPS)
    get_path_mock.assert_called_once_with(self.CON_PROPS)
    self.assertEqual(get_path_mock.return_value, res)
    self.assertEqual([], self.cmds)