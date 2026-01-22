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
def test_get_volume_paths_replicated(self):
    """Search for device from replicated conn props with >1 replica."""
    conn_props = nvmeof.NVMeOFConnProps(connection_properties)
    self.assertEqual(['/dev/md/fakealias'], self.connector.get_volume_paths(conn_props))