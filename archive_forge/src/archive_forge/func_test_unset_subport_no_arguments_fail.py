import copy
from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
import testtools
from openstackclient.network.v2 import network_trunk
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
def test_unset_subport_no_arguments_fail(self):
    arglist = [self._trunk['name']]
    verifylist = [('trunk', self._trunk['name'])]
    self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)