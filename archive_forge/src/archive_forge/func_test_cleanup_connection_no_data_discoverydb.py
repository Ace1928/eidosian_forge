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
def test_cleanup_connection_no_data_discoverydb(self):
    self.connector.use_multipath = True
    with mock.patch.object(self.connector, '_get_discoverydb_portals', side_effect=exception.TargetPortalsNotFound), mock.patch.object(self.connector._linuxscsi, 'remove_connection') as mock_remove:
        self.connector._cleanup_connection(self.SINGLE_CON_PROPS)
        mock_remove.assert_not_called()