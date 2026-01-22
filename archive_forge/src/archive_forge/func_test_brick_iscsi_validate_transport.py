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
@mock.patch.object(iscsi.ISCSIConnector, '_run_iscsiadm_bare')
def test_brick_iscsi_validate_transport(self, mock_iscsiadm):
    sample_output = '# BEGIN RECORD 2.0-872\niface.iscsi_ifacename = %s.fake_suffix\niface.net_ifacename = <empty>\niface.ipaddress = <empty>\niface.hwaddress = 00:53:00:00:53:00\niface.transport_name = %s\niface.initiatorname = <empty>\n# END RECORD'
    for tport in self.connector.supported_transports:
        mock_iscsiadm.return_value = (sample_output % (tport, tport), '')
        self.assertEqual(tport + '.fake_suffix', self.connector._validate_iface_transport(tport + '.fake_suffix'))
    mock_iscsiadm.return_value = ('', 'iscsiadm: Could not read iface fake_transport (6)')
    self.assertEqual('default', self.connector._validate_iface_transport('fake_transport'))