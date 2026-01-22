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
def test_discover_iscsi_portals(self):
    location = '10.0.2.15:3260'
    name = 'volume-00000001'
    iqn = 'iqn.2010-10.org.openstack:%s' % name
    vol = {'id': 1, 'name': name}
    auth_method = 'CHAP'
    auth_username = 'fake_chap_username'
    auth_password = 'fake_chap_password'
    discovery_auth_method = 'CHAP'
    discovery_auth_username = 'fake_chap_username'
    discovery_auth_password = 'fake_chap_password'
    connection_properties = self.iscsi_connection_chap(vol, location, iqn, auth_method, auth_username, auth_password, discovery_auth_method, discovery_auth_username, discovery_auth_password)
    self.connector_with_multipath = iscsi.ISCSIConnector(None, execute=self.fake_execute, use_multipath=True)
    for transport in ['default', 'iser', 'badTransport']:
        interface = 'iser' if transport == 'iser' else 'default'
        self.mock_object(self.connector_with_multipath, '_get_transport', mock.Mock(return_value=interface))
        self.connector_with_multipath._discover_iscsi_portals(connection_properties['data'])
        expected_cmds = ['iscsiadm -m discoverydb -t sendtargets -I %(iface)s -p %(location)s --op update -n discovery.sendtargets.auth.authmethod -v %(auth_method)s -n discovery.sendtargets.auth.username -v %(username)s -n discovery.sendtargets.auth.password -v %(password)s' % {'iface': interface, 'location': location, 'auth_method': discovery_auth_method, 'username': discovery_auth_username, 'password': discovery_auth_password}, 'iscsiadm -m node --op show -p %s' % location, 'iscsiadm -m discoverydb -t sendtargets -I %(iface)s -p %(location)s --discover' % {'iface': interface, 'location': location}, 'iscsiadm -m node --op show -p %s' % location]
        self.assertEqual(expected_cmds, self.cmds)
        self.cmds = list()