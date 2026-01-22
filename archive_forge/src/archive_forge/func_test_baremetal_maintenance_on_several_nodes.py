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
def test_baremetal_maintenance_on_several_nodes(self):
    arglist = ['node_uuid', 'node_name']
    verifylist = [('nodes', ['node_uuid', 'node_name'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.baremetal_mock.node.set_maintenance.assert_has_calls([mock.call(n, True, maint_reason=None) for n in arglist])