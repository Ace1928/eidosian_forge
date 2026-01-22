from unittest import mock
from osc_lib import exceptions
import testtools
from neutronclient.osc.v2.sfc import sfc_port_chain
from neutronclient.tests.unit.osc.v2.sfc import fakes
def test_set_only_no_port_pair_group(self):
    target = self.resource['id']
    arglist = [target, '--no-port-pair-group']
    verifylist = [(self.res, target), ('no_port_pair_group', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)