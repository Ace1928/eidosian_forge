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
def test_sanitize_log_run_iscsiadm(self):

    def fake_debug(*args, **kwargs):
        self.assertIn('node.session.auth.password', args[0])
        self.assertNotIn('scrubme', args[0])
    volume = {'id': 'fake_uuid'}
    connection_info = self.iscsi_connection(volume, '10.0.2.15:3260', 'fake_iqn')
    iscsi_properties = connection_info['data']
    with mock.patch.object(iscsi.LOG, 'debug', side_effect=fake_debug) as debug_mock:
        self.connector._iscsiadm_update(iscsi_properties, 'node.session.auth.password', 'scrubme')
        self.assertTrue(debug_mock.called)