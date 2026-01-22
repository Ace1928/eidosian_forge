from unittest import mock
from osc_lib import utils
from troveclient import common
from troveclient import exceptions
from troveclient.osc.v1 import database_configurations
from troveclient.tests.osc.v1 import fakes
def test_configuration_list_defaults(self):
    parsed_args = self.check_parser(self.cmd, [], [])
    columns, data = self.cmd.take_action(parsed_args)
    self.configuration_client.list.assert_called_once_with(**self.defaults)
    self.assertEqual(self.columns, columns)
    self.assertEqual([tuple(self.values)], data)