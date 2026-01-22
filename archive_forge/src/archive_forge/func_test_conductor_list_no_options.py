import copy
from osc_lib.tests import utils as oscutils
from ironicclient.osc.v1 import baremetal_conductor
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_conductor_list_no_options(self):
    arglist = []
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'marker': None, 'limit': None}
    self.baremetal_mock.conductor.list.assert_called_with(**kwargs)
    collist = ('Hostname', 'Conductor Group', 'Alive')
    self.assertEqual(collist, columns)
    datalist = ((baremetal_fakes.baremetal_hostname, baremetal_fakes.baremetal_conductor_group, baremetal_fakes.baremetal_alive),)
    self.assertEqual(datalist, tuple(data))