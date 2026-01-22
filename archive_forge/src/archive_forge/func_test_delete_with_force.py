import pkg_resources as pkg
from unittest import mock
from oslo_serialization import jsonutils
from mistralclient.api.v2 import executions
from mistralclient.commands.v2 import executions as execution_cmd
from mistralclient.tests.unit import base
def test_delete_with_force(self):
    self.call(execution_cmd.Delete, app_args=['id', '--force'])
    self.client.executions.delete.assert_called_once_with('id', force=True)