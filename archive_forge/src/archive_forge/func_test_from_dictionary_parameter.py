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
@mock.patch.object(nvmeof.Target, 'factory')
def test_from_dictionary_parameter(self, mock_target):
    """Decorator converts dict into connection properties instance."""

    class Connector(object):

        @nvmeof.NVMeOFConnProps.from_dictionary_parameter
        def connect_volume(my_self, connection_properties):
            self.assertIsInstance(connection_properties, nvmeof.NVMeOFConnProps)
            return 'result'
    conn = Connector()
    conn_props = {'target_nqn': 'nqn_value', 'vol_uuid': 'uuid', 'portals': [('portal1', 'port_value', 'RoCEv2'), ('portal2', 'port_value', 'anything')]}
    res = conn.connect_volume(conn_props)
    self.assertEqual('result', res)