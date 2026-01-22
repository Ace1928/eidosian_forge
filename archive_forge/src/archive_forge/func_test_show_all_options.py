from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network_segment_range
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_show_all_options(self):
    arglist = [self._network_segment_range.id]
    verifylist = [('network_segment_range', self._network_segment_range.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.find_network_segment_range.assert_called_once_with(self._network_segment_range.id, ignore_missing=False)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, data)