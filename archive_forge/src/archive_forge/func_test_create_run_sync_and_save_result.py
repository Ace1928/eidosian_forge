import copy
import io
import sys
from unittest import mock
from oslo_serialization import jsonutils
from mistralclient.api.v2 import action_executions as action_ex
from mistralclient.commands.v2 import action_executions as action_ex_cmd
from mistralclient.tests.unit import base
def test_create_run_sync_and_save_result(self):
    self.client.action_executions.create.return_value = ACTION_EX_WITH_OUTPUT
    self.call(action_ex_cmd.Create, app_args=['some', '{"output": "Hello!"}', '--save-result', '--run-sync'])
    self.assertDictEqual(ACTION_EX_RESULT, jsonutils.loads(self.app.stdout.write.call_args[0][0]))