from unittest import mock
from magnumclient.common.apiclient import exceptions
from magnumclient.tests.v1 import shell_test_base
from magnumclient.v1.cluster_templates import ClusterTemplate
@mock.patch('magnumclient.v1.cluster_templates.ClusterTemplateManager.update')
def test_cluster_template_update_failure_few_args(self, mock_update):
    _error_msg = ['.*?^usage: magnum cluster-template-update ', '.*?^error: (the following arguments|too few arguments)', ".*?^Try 'magnum help cluster-template-update' for more information."]
    self._test_arg_failure('cluster-template-update', _error_msg)
    mock_update.assert_not_called()
    self._test_arg_failure('cluster-template-update test', _error_msg)
    mock_update.assert_not_called()
    self._test_arg_failure('cluster-template-update test add', _error_msg)
    mock_update.assert_not_called()