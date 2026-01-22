import builtins
import errno
import os.path
from unittest import mock
import ddt
from oslo_concurrency import processutils as putils
from os_brick import exception
from os_brick import executor
from os_brick.initiator.connectors import nvmeof
from os_brick.privileged import nvmeof as priv_nvmeof
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests import base as test_base
from os_brick.tests.initiator import test_connector
from os_brick import utils
@mock.patch.object(nvmeof.Target, '_get_nvme_devices')
@mock.patch.object(nvmeof.Target, 'set_portals_controllers')
@mock.patch.object(nvmeof.Portal, '__init__', return_value=None)
def test_init_no_id(self, mock_init, mock_set_ctrls, mock_get_devs):
    """With no ID parameters query existing nvme devices."""
    target = nvmeof.Target(self.conn_props, 'nqn', self.conn_props_dict['portals'])
    self.assertEqual(self.conn_props, target.source_conn_props)
    self.assertEqual('nqn', target.nqn)
    for name in ('uuid', 'nguid', 'ns_id'):
        self.assertIsNone(getattr(target, name))
    self.assertIsInstance(target.portals[0], nvmeof.Portal)
    self.assertIsInstance(target.portals[1], nvmeof.Portal)
    mock_set_ctrls.assert_not_called()
    mock_get_devs.assert_called_once_with()
    self.assertEqual(2, mock_init.call_count)
    mock_init.assert_has_calls([mock.call(target, 'portal1', 'port1', 'RoCEv2'), mock.call(target, 'portal2', 'port2', 'anything')])