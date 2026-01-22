import copy
import io
import json
import sys
from unittest import mock
from osc_lib.tests import utils as oscutils
from ironicclient.common import utils as commonutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_node
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
from ironicclient.v1 import utils as v1_utils
def test_baremetal_node_inventory_save(self):
    arglist = ['node_uuid']
    verifylist = [('node', 'node_uuid')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    buf = io.StringIO()
    with mock.patch.object(sys, 'stdout', buf):
        self.cmd.take_action(parsed_args)
    self.baremetal_mock.node.get_inventory.assert_called_once_with('node_uuid')
    expected_data = {'memory': {'physical_mb': 3072}, 'cpu': {'count': 1, 'model_name': 'qemu64', 'architecture': 'x86_64'}, 'disks': [{'name': 'testvm2.qcow2', 'size': 11811160064}], 'interfaces': [{'mac_address': '52:54:00:11:2d:26'}], 'system_vendor': {'product_name': 'testvm2', 'manufacturer': 'Sushy Emulator'}, 'boot': {'current_boot_mode': 'uefi'}}
    inventory = json.loads(buf.getvalue())
    self.assertEqual(expected_data, inventory['inventory'])