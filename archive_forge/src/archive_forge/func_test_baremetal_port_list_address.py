import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from osc_lib import utils as oscutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_port
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_port_list_address(self):
    arglist = ['--address', baremetal_fakes.baremetal_port_address]
    verifylist = [('address', baremetal_fakes.baremetal_port_address)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    kwargs = {'address': baremetal_fakes.baremetal_port_address, 'marker': None, 'limit': None}
    self.baremetal_mock.port.list.assert_called_with(**kwargs)