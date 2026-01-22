from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.network.v2 import port
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
def test_set_port_invalid_data_plane_status_value(self):
    arglist = ['--data-plane-status', 'Spider-Man', 'test-port']
    self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, None)