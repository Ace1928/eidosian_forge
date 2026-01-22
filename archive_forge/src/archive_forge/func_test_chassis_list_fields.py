import copy
from unittest import mock
from osc_lib.tests import utils as oscutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_chassis
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_chassis_list_fields(self):
    arglist = ['--fields', 'uuid', 'extra']
    verifylist = [('fields', [['uuid', 'extra']])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    kwargs = {'marker': None, 'limit': None, 'detail': False, 'fields': ('uuid', 'extra')}
    self.baremetal_mock.chassis.list.assert_called_with(**kwargs)