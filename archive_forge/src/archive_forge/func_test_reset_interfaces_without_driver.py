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
def test_reset_interfaces_without_driver(self):
    arglist = ['node_uuid', '--reset-interfaces']
    verifylist = [('nodes', ['node_uuid']), ('reset_interfaces', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)
    self.assertFalse(self.baremetal_mock.node.update.called)