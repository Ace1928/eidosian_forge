from unittest import mock
from unittest.mock import call
from cinderclient import api_versions
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_transfer_request
def test_transfer_delete(self):
    arglist = [self.volume_transfers[0].id]
    verifylist = [('transfer_request', [self.volume_transfers[0].id])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.transfer_mock.delete.assert_called_with(self.volume_transfers[0].id)
    self.assertIsNone(result)