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
def test_baremetal_unset_multiple_properties(self):
    arglist = ['node_uuid', '--property', 'path/to/property', '--property', 'other/path']
    verifylist = [('nodes', ['node_uuid']), ('property', ['path/to/property', 'other/path'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/properties/path/to/property', 'op': 'remove'}, {'path': '/properties/other/path', 'op': 'remove'}])