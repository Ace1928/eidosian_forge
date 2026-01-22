from unittest import mock
from magnumclient.tests.v1 import shell_test_base
from magnumclient.v1.stats import Stats
@mock.patch('magnumclient.v1.stats.StatsManager.list')
def test_stats_get_success(self, mock_list):
    self._test_arg_success('stats-list')
    expected_args = self._get_expected_args_list()
    mock_list.assert_called_once_with(**expected_args)