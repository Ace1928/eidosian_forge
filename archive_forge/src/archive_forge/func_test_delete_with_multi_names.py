import pkg_resources as pkg
from unittest import mock
from oslo_serialization import jsonutils
from mistralclient.api.v2 import executions
from mistralclient.commands.v2 import executions as execution_cmd
from mistralclient.tests.unit import base
def test_delete_with_multi_names(self):
    self.call(execution_cmd.Delete, app_args=['id1', 'id2'])
    self.assertEqual(2, self.client.executions.delete.call_count)
    self.assertEqual([mock.call('id1', force=False), mock.call('id2', force=False)], self.client.executions.delete.call_args_list)