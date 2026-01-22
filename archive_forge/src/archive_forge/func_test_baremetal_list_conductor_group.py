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
def test_baremetal_list_conductor_group(self):
    conductor_group = 'in-the-closet-to-the-left'
    arglist = ['--conductor-group', conductor_group]
    verifylist = [('conductor_group', conductor_group)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    kwargs = {'marker': None, 'limit': None, 'conductor_group': conductor_group}
    self.baremetal_mock.node.list.assert_called_with(**kwargs)