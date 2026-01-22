from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_network_create_missing_options(self, net_mock):
    net_mock.return_value = self._network
    arglist = [self._network['label']]
    verifylist = [('name', self._network['label'])]
    self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)