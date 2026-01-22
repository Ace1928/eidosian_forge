from unittest import mock
from osc_lib import utils
from troveclient import common
from troveclient import exceptions
from troveclient.osc.v1 import database_configurations
from troveclient.tests.osc.v1 import fakes
def test_configuration_parameters_list_defaults(self):
    args = ['d-123', '--datastore', 'mysql']
    verifylist = [('datastore_version', 'd-123'), ('datastore', 'mysql')]
    parsed_args = self.check_parser(self.cmd, args, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.assertEqual(self.columns, columns)
    self.assertEqual([tuple(self.values)], data)