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
def test_baremetal_show_fields_multiple(self):
    arglist = ['xxxxx', '--fields', 'uuid', 'name', '--fields', 'extra']
    verifylist = [('node', 'xxxxx'), ('fields', [['uuid', 'name'], ['extra']])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.assertNotIn('chassis_uuid', columns)
    args = ['xxxxx']
    fields = ['uuid', 'name', 'extra']
    self.baremetal_mock.node.get.assert_called_with(*args, fields=fields)