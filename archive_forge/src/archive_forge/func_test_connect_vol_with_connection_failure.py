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
@mock.patch.object(iscsi.ISCSIConnector, '_connect_to_iscsi_portal')
def test_connect_vol_with_connection_failure(self, connect_mock):
    data = self._get_connect_vol_data()
    connect_mock.side_effect = Exception()
    self.connector._connect_vol(3, self.CON_PROPS, data)
    expected = self._get_connect_vol_data()
    expected.update(failed_logins=1, stopped_threads=1)
    self.assertDictEqual(expected, data)