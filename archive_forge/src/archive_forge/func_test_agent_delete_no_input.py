from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.compute.v2 import agent
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_agent_delete_no_input(self):
    arglist = []
    verifylist = None
    self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)