import operator
from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from osc_lib.utils import columns as column_util
from neutronclient.tests.unit.osc.v2.networking_bgpvpn import fakes
def test_delete_one_association(self):
    fake_bgpvpn = fakes.create_one_bgpvpn()
    fake_res = fakes.create_one_resource()
    fake_res_assoc = fakes.create_one_resource_association(fake_res)
    self.networkclient.delete_bgpvpn_router_association = mock.Mock()
    arglist = [fake_res_assoc['id'], fake_bgpvpn['id']]
    verifylist = [('resource_association_ids', [fake_res_assoc['id']]), ('bgpvpn', fake_bgpvpn['id'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.networkclient.delete_bgpvpn_router_association.assert_called_once_with(fake_bgpvpn['id'], fake_res_assoc['id'])
    self.assertIsNone(result)