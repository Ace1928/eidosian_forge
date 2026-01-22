from unittest import mock
from unittest.mock import call
from cinderclient import api_versions
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_transfer_request
def test_transfer_accept_no_option(self):
    arglist = [self.volume_transfer.id]
    verifylist = [('transfer_request', self.volume_transfer.id)]
    self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)