from unittest import mock
from unittest.mock import call
from cinderclient import api_versions
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_transfer_request
def test_transfer_list_without_argument(self):
    arglist = []
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    expected_columns = ['ID', 'Name', 'Volume']
    self.assertEqual(expected_columns, columns)
    datalist = ((self.volume_transfers.id, self.volume_transfers.name, self.volume_transfers.volume_id),)
    self.assertEqual(datalist, tuple(data))
    self.transfer_mock.list.assert_called_with(detailed=True, search_opts={'all_tenants': 0})