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
@mock.patch.object(utils, 'get_nvme_host_id', return_value=SYS_UUID)
@mock.patch.object(nvmeof.NVMeOFConnector, '_is_native_multipath_supported', return_value=True)
@mock.patch.object(nvmeof.NVMeOFConnector, 'nvme_present')
@mock.patch.object(utils, 'get_host_nqn', autospec=True)
@mock.patch.object(priv_nvmeof, 'get_system_uuid', autospec=True)
@mock.patch.object(nvmeof.NVMeOFConnector, '_get_host_uuid', autospec=True)
def test_get_connector_properties_with_sysuuid(self, mock_host_uuid, mock_sysuuid, mock_nqn, mock_nvme_present, mock_native_mpath_support, mock_get_host_id):
    mock_host_uuid.return_value = HOST_UUID
    mock_sysuuid.return_value = SYS_UUID
    mock_nqn.return_value = HOST_NQN
    mock_nvme_present.return_value = True
    props = self.connector.get_connector_properties('sudo')
    expected_props = {'system uuid': SYS_UUID, 'nqn': HOST_NQN, 'uuid': HOST_UUID, 'nvme_native_multipath': False, 'nvme_hostid': SYS_UUID}
    self.assertEqual(expected_props, props)
    mock_get_host_id.assert_called_once_with(SYS_UUID)
    mock_nqn.assert_called_once_with(SYS_UUID)