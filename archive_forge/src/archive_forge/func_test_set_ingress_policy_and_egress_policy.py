import copy
import re
from unittest import mock
from osc_lib import exceptions
from osc_lib.tests import utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.fwaas import firewallgroup
from neutronclient.osc.v2 import utils as v2_utils
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.fwaas import common
from neutronclient.tests.unit.osc.v2.fwaas import fakes
def test_set_ingress_policy_and_egress_policy(self):
    target = self.resource['id']
    ingress_policy = 'ingress_policy'
    egress_policy = 'egress_policy'

    def _mock_fwg_policy(*args, **kwargs):
        if self.networkclient.find_firewall_group.call_count == 1:
            self.networkclient.find_firewall_group.assert_called_with(target)
        if self.networkclient.find_firewall_policy.call_count == 1:
            self.networkclient.find_firewall_policy.assert_called_with(ingress_policy)
        if self.networkclient.find_firewall_policy.call_count == 2:
            self.networkclient.find_firewall_policy.assert_called_with(egress_policy)
        return {'id': args[0]}
    self.networkclient.find_firewall_group.side_effect = _mock_fwg_policy
    self.networkclient.find_firewall_policy.side_effect = _mock_fwg_policy
    arglist = [target, '--ingress-firewall-policy', ingress_policy, '--egress-firewall-policy', egress_policy]
    verifylist = [(self.res, target), ('ingress_firewall_policy', ingress_policy), ('egress_firewall_policy', egress_policy)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.mocked.assert_called_once_with(target, **{'ingress_firewall_policy_id': ingress_policy, 'egress_firewall_policy_id': egress_policy})
    self.assertIsNone(result)