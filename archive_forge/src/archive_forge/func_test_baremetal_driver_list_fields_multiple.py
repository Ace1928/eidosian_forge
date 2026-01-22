import copy
from osc_lib.tests import utils as oscutils
from ironicclient.osc.v1 import baremetal_driver
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_driver_list_fields_multiple(self):
    arglist = ['--fields', 'name', '--fields', 'hosts', 'type']
    verifylist = [('fields', [['name'], ['hosts', 'type']])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    kwargs = {'driver_type': None, 'detail': None, 'fields': ('name', 'hosts', 'type')}
    self.baremetal_mock.driver.list.assert_called_with(**kwargs)