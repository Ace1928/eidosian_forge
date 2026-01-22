import operator
from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from osc_lib.utils import columns as column_util
from neutronclient.tests.unit.osc.v2.networking_bgpvpn import fakes
def test_set_resource_association(self):
    fake_bgpvpn = fakes.create_one_bgpvpn()
    fake_res = fakes.create_one_resource()
    fake_res_assoc = fakes.create_one_resource_association(fake_res)
    self.networkclient.update_bgpvpn_router_association = mock.Mock(return_value={fakes.BgpvpnFakeAssoc._resource: fake_res_assoc})
    arglist = [fake_res_assoc['id'], fake_bgpvpn['id']]
    verifylist = [('resource_association_id', fake_res_assoc['id']), ('bgpvpn', fake_bgpvpn['id'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.networkclient.update_bgpvpn_router_association.assert_not_called()
    self.assertIsNone(result)