import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from osc_lib import utils as oscutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_port
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_port_show_address(self):
    arglist = ['--address', baremetal_fakes.baremetal_port_address]
    verifylist = [('address', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    args = {'AA:BB:CC:DD:EE:FF'}
    self.baremetal_mock.port.get_by_address.assert_called_with(*args, fields=None)