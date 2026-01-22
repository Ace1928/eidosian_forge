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
@mock.patch.object(iscsi.ISCSIConnector, '_get_node_startup_values')
@mock.patch.object(iscsi.ISCSIConnector, '_iscsiadm_update')
def test_recover_node_startup_values(self, iscsiadm_update_mock, get_node_startup_values_mock):
    name1 = 'volume-00000001-1'
    name2 = 'volume-00000001-2'
    name3 = 'volume-00000001-3'
    vol = {'id': 1, 'name': name1}
    location = '10.0.2.15:3260'
    iqn1 = 'iqn.2010-10.org.openstack:%s' % name1
    iqn2 = 'iqn.2010-10.org.openstack:%s' % name2
    iqn3 = 'iqn.2010-10.org.openstack:%s' % name3
    connection_properties = self.iscsi_connection(vol, [location], iqn1)
    recover_connection = self.iscsi_connection(vol, [location], iqn2)
    node_startup1 = 'manual'
    node_startup2 = 'automatic'
    node_startup3 = 'manual'
    get_node_startup_values_mock.return_value = {iqn1: node_startup1, iqn2: node_startup2, iqn3: node_startup3}
    old_node_startup_values = {iqn1: node_startup1, iqn2: 'manual', iqn3: node_startup3}
    self.connector._recover_node_startup_values(connection_properties['data'], old_node_startup_values)
    iscsiadm_update_mock.assert_called_once_with(recover_connection['data'], 'node.startup', 'manual')