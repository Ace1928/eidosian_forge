import operator
from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from osc_lib.utils import columns as column_util
from neutronclient.tests.unit.osc.v2.networking_bgpvpn import fakes
def test_delete_multi_bpgvpn(self):
    count = 3
    fake_bgpvpn = fakes.create_one_bgpvpn()
    fake_res = fakes.create_resources(count=count)
    fake_res_assocs = fakes.create_resource_associations(fake_res)
    fake_res_assoc_ids = [fake_res_assoc['id'] for fake_res_assoc in fake_res_assocs]
    self.networkclient.delete_bgpvpn_router_association = mock.Mock()
    arglist = fake_res_assoc_ids + [fake_bgpvpn['id']]
    verifylist = [('resource_association_ids', fake_res_assoc_ids), ('bgpvpn', fake_bgpvpn['id'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.networkclient.delete_bgpvpn_router_association.assert_has_calls([mock.call(fake_bgpvpn['id'], id) for id in fake_res_assoc_ids])
    self.assertIsNone(result)