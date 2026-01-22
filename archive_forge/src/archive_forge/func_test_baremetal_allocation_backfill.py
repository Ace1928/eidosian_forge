import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_allocation
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_allocation_backfill(self):
    arglist = ['--node', baremetal_fakes.baremetal_uuid]
    verifylist = [('node', baremetal_fakes.baremetal_uuid)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    args = {'node': baremetal_fakes.baremetal_uuid}
    self.baremetal_mock.allocation.create.assert_called_once_with(**args)