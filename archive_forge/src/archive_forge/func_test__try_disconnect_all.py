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
@mock.patch.object(nvmeof.NVMeOFConnector, '_try_disconnect')
@mock.patch.object(nvmeof.Target, 'set_portals_controllers')
def test__try_disconnect_all(self, mock_set_portals, mock_disconnect):
    """Disconnect all portals for all targets in connection properties."""
    connection_properties = {'vol_uuid': VOL_UUID, 'alias': 'raid_alias', 'replica_count': 2, 'volume_replicas': [{'target_nqn': 'nqn1', 'vol_uuid': VOL_UUID1, 'portals': [['portal1', 'port_value', 'RoCEv2'], ['portal2', 'port_value', 'anything']]}, {'target_nqn': 'nqn2', 'vol_uuid': VOL_UUID2, 'portals': [['portal4', 'port_value', 'anything'], ['portal3', 'port_value', 'RoCEv2']]}]}
    conn_props = nvmeof.NVMeOFConnProps(connection_properties)
    exc = exception.ExceptionChainer()
    self.connector._try_disconnect_all(conn_props, exc)
    self.assertEqual(2, mock_set_portals.call_count)
    mock_set_portals.assert_has_calls((mock.call(), mock.call()))
    self.assertEqual(4, mock_disconnect.call_count)
    mock_disconnect.assert_has_calls((mock.call(conn_props.targets[0].portals[0]), mock.call(conn_props.targets[0].portals[1]), mock.call(conn_props.targets[1].portals[0]), mock.call(conn_props.targets[1].portals[1])))
    self.assertFalse(bool(exc))