import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient.osc.v1 import baremetal_portgroup
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_portgroup_list_address(self):
    arglist = ['--address', baremetal_fakes.baremetal_portgroup_address]
    verifylist = [('address', baremetal_fakes.baremetal_portgroup_address)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'address': baremetal_fakes.baremetal_portgroup_address, 'marker': None, 'limit': None}
    self.baremetal_mock.portgroup.list.assert_called_with(**kwargs)
    collist = ('UUID', 'Address', 'Name')
    self.assertEqual(collist, columns)
    datalist = ((baremetal_fakes.baremetal_portgroup_uuid, baremetal_fakes.baremetal_portgroup_address, baremetal_fakes.baremetal_portgroup_name),)
    self.assertEqual(datalist, tuple(data))