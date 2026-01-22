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
def test_child_node_list(self):
    arglist = ['node_uuid']
    verifylist = [('node', 'node_uuid')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.baremetal_mock.node.list_children_of_node.assert_called_once_with('node_uuid')
    self.assertEqual(('Child Nodes',), columns)
    self.assertEqual([[node] for node in baremetal_fakes.CHILDREN], data)