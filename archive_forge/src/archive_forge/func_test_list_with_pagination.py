import pkg_resources as pkg
from unittest import mock
from oslo_serialization import jsonutils
from mistralclient.api.v2 import executions
from mistralclient.commands.v2 import executions as execution_cmd
from mistralclient.tests.unit import base
def test_list_with_pagination(self):
    self.client.executions.list.return_value = [EXEC]
    self.call(execution_cmd.List)
    self.client.executions.list.assert_called_once_with(fields=execution_cmd.ExecutionFormatter.fields(), limit=100, marker='', nulls='', sort_dirs='desc', sort_keys='created_at', task=None)
    self.call(execution_cmd.List, app_args=['--oldest'])
    self.client.executions.list.assert_called_with(fields=execution_cmd.ExecutionFormatter.fields(), limit=100, marker='', nulls='', sort_keys='created_at', sort_dirs='asc', task=None)
    self.call(execution_cmd.List, app_args=['--limit', '5', '--sort_keys', 'id, Workflow', '--sort_dirs', 'desc', '--marker', 'abc'])
    self.client.executions.list.assert_called_with(fields=execution_cmd.ExecutionFormatter.fields(), limit=5, marker='abc', nulls='', sort_keys='id, Workflow', sort_dirs='desc', task=None)