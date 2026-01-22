import copy
from unittest import mock
from osc_lib import exceptions
from osc_lib.tests import utils
import testtools
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.logging import network_log
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.logging import fakes
def test_illegal_set_resource_type(self):
    target = self.res['id']
    resource_type = 'security_group'
    arglist = [target, '--resource-type', resource_type]
    verifylist = [('network_log', target), ('resource_type', resource_type)]
    self.assertRaises(utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)