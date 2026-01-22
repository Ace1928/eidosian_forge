import re
from unittest import mock
from osc_lib import exceptions
from osc_lib.tests import utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.fwaas import firewallpolicy
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.fwaas import common
from neutronclient.tests.unit.osc.v2.fwaas import fakes
def test_create_with_firewall_rule_and_no(self):
    name = 'my-fwp'
    rule1 = 'rule1'
    rule2 = 'rule2'
    arglist = [name, '--firewall-rule', rule1, '--firewall-rule', rule2, '--no-firewall-rule']
    verifylist = [('name', name), ('firewall_rule', [rule1, rule2]), ('no_firewall_rule', True)]
    self.assertRaises(utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)