from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import floating_ip as fip
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_floating_ip_create_no_arg(self, fip_mock):
    arglist = []
    verifylist = []
    self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)