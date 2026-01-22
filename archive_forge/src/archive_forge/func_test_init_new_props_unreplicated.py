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
@ddt.data('vol_uuid', 'ns_id', 'volume_nguid')
@mock.patch.object(nvmeof.Target, 'factory')
def test_init_new_props_unreplicated(self, id_name, mock_target):
    """Test init with new format connection properties but no replicas."""
    conn_props = {'target_nqn': 'nqn_value', id_name: 'uuid', 'portals': [('portal1', 'port_value', 'RoCEv2'), ('portal2', 'port_value', 'anything')], 'qos_specs': None, 'access_mode': 'rw', 'encrypted': False, 'cacheable': True, 'discard': True}
    res = nvmeof.NVMeOFConnProps(conn_props, mock.sentinel.find_controllers)
    self.assertFalse(res.is_replicated)
    self.assertIsNone(res.qos_specs)
    self.assertFalse(res.readonly)
    self.assertFalse(res.encrypted)
    self.assertTrue(res.cacheable)
    self.assertTrue(res.discard)
    self.assertIsNone(res.alias)
    self.assertIsNone(res.cinder_volume_id)
    kw_id_arg = {id_name: 'uuid'}
    mock_target.assert_called_once_with(source_conn_props=res, find_controllers=mock.sentinel.find_controllers, target_nqn='nqn_value', portals=[('portal1', 'port_value', 'RoCEv2'), ('portal2', 'port_value', 'anything')], qos_specs=None, access_mode='rw', encrypted=False, cacheable=True, discard=True, **kw_id_arg)
    self.assertListEqual([mock_target.return_value], res.targets)