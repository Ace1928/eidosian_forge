import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from osc_lib import utils as oscutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_port
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_port_list_node(self):
    arglist = ['--node', baremetal_fakes.baremetal_uuid]
    verifylist = [('node', baremetal_fakes.baremetal_uuid)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    kwargs = {'node': baremetal_fakes.baremetal_uuid, 'marker': None, 'limit': None}
    self.baremetal_mock.port.list.assert_called_with(**kwargs)