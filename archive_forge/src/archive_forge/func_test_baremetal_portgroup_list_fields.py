import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient.osc.v1 import baremetal_portgroup
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_portgroup_list_fields(self):
    arglist = ['--fields', 'uuid', 'address']
    verifylist = [('fields', [['uuid', 'address']])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    kwargs = {'marker': None, 'limit': None, 'detail': False, 'fields': ('uuid', 'address')}
    self.baremetal_mock.portgroup.list.assert_called_with(**kwargs)