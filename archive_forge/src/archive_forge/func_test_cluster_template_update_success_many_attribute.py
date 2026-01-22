from unittest import mock
from magnumclient.common.apiclient import exceptions
from magnumclient.tests.v1 import shell_test_base
from magnumclient.v1.cluster_templates import ClusterTemplate
@mock.patch('magnumclient.v1.cluster_templates.ClusterTemplateManager.update')
def test_cluster_template_update_success_many_attribute(self, mock_update):
    self._test_arg_success('cluster-template-update test add test=test test1=test1')
    patch = [{'op': 'add', 'path': '/test', 'value': 'test'}, {'op': 'add', 'path': '/test1', 'value': 'test1'}]
    mock_update.assert_called_once_with('test', patch)