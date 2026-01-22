import re
from unittest import mock
from osc_lib import exceptions
from osc_lib.tests import utils
import testtools
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.fwaas import firewallrule
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.fwaas import common
from neutronclient.tests.unit.osc.v2.fwaas import fakes
def test_set_destination_port_and_no(self):
    target = self.resource['id']
    arglist = [target, '--destination-port', '1:54321', '--no-destination-port']
    verifylist = [(self.res, target), ('destination_port', '1:54321'), ('no_destination_port', True)]
    self.assertRaises(utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)