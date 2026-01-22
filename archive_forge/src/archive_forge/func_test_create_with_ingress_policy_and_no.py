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
def test_create_with_ingress_policy_and_no(self):
    policy = 'my-policy'
    arglist = ['--ingress-firewall-policy', policy, '--no-ingress-firewall-policy']
    verifylist = [('ingress_firewall_policy', policy), ('no_ingress_firewall_policy', True)]
    self.assertRaises(utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)