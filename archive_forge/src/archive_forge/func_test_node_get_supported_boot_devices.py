import copy
import tempfile
import time
from unittest import mock
import testtools
from testtools.matchers import HasLength
from ironicclient.common import utils as common_utils
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import node
from ironicclient.v1 import volume_connector
from ironicclient.v1 import volume_target
def test_node_get_supported_boot_devices(self):
    boot_device = self.mgr.get_supported_boot_devices(NODE1['uuid'])
    expect = [('GET', '/v1/nodes/%s/management/boot_device/supported' % NODE1['uuid'], {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(SUPPORTED_BOOT_DEVICE, boot_device)