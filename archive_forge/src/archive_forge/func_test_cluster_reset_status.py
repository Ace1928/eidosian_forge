from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from troveclient import common
from troveclient.osc.v1 import database_clusters
from troveclient.tests.osc.v1 import fakes
@mock.patch.object(utils, 'find_resource')
def test_cluster_reset_status(self, mock_find):
    args = ['cluster1']
    mock_find.return_value = args[0]
    parsed_args = self.check_parser(self.cmd, args, [])
    result = self.cmd.take_action(parsed_args)
    self.cluster_client.reset_status.assert_called_with('cluster1')
    self.assertIsNone(result)