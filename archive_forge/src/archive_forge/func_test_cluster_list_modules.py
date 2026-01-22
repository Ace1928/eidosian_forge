from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from troveclient import common
from troveclient.osc.v1 import database_clusters
from troveclient.tests.osc.v1 import fakes
@mock.patch.object(utils, 'find_resource')
def test_cluster_list_modules(self, mock_find_resource):
    mock_find_resource.return_value = self.data
    args = ['cls-1234']
    parsed_args = self.check_parser(self.cmd, args, [])
    columns, data = self.cmd.take_action(parsed_args)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.values, data)