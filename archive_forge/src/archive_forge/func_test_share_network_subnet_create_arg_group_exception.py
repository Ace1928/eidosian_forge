from unittest import mock
import uuid
import ddt
from osc_lib import exceptions
from manilaclient import api_versions
from manilaclient.osc.v2 import share_network_subnets as osc_share_subnets
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_network_subnet_create_arg_group_exception(self):
    fake_neutron_net_id = str(uuid.uuid4())
    arglist = [self.share_network.id, '--neutron-net-id', fake_neutron_net_id]
    verifylist = [('share_network', self.share_network.id), ('neutron_net_id', fake_neutron_net_id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)