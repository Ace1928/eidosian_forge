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
def test_baremetal_set_chassis(self):
    chassis = '4f4135ea-7e58-4e3d-bcc4-b87ca16e980b'
    arglist = ['node_uuid', '--chassis-uuid', chassis]
    verifylist = [('nodes', ['node_uuid']), ('chassis_uuid', chassis)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/chassis_uuid', 'value': chassis, 'op': 'add'}], reset_interfaces=None)