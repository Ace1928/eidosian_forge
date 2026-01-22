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
def test_rescue_baremetal_no_wait(self):
    arglist = ['node_uuid', '--rescue-password', 'supersecret']
    verifylist = [('nodes', ['node_uuid']), ('provision_state', 'rescue'), ('rescue_password', 'supersecret')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.baremetal_mock.node.set_provision_state.assert_called_once_with('node_uuid', 'rescue', cleansteps=None, deploysteps=None, configdrive=None, rescue_password='supersecret')