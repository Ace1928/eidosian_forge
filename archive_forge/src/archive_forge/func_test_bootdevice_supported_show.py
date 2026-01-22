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
def test_bootdevice_supported_show(self):
    arglist = ['node_uuid', '--supported']
    verifylist = [('node', 'node_uuid'), ('supported', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    mock = self.baremetal_mock.node.get_supported_boot_devices
    mock.assert_called_once_with('node_uuid')