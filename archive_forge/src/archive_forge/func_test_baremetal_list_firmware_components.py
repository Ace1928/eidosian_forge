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
def test_baremetal_list_firmware_components(self):
    arglist = ['node_uuid']
    verifylist = [('node', 'node_uuid')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.baremetal_mock.node.list_firmware_components.assert_called_once_with('node_uuid')
    expected_columns = ('Component', 'Initial Version', 'Current Version', 'Last Version Flashed', 'Created At', 'Updated At')
    self.assertEqual(expected_columns, columns)
    expected_data = [(fw['component'], fw['initial_version'], fw['current_version'], fw['last_version_flashed'], fw['created_at'], fw['updated_at']) for fw in baremetal_fakes.FIRMWARE_COMPONENTS]
    self.assertEqual(tuple(expected_data), tuple(data))