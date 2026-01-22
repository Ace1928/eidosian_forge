from unittest import mock
from oslo_serialization import jsonutils
from mistralclient.api.v2.executions import Execution
from mistralclient.api.v2 import tasks
from mistralclient.commands.v2 import tasks as task_cmd
from mistralclient.tests.unit import base
def test_rerun_no_reset_update_env(self):
    self.client.tasks.rerun.return_value = TASK
    result = self.call(task_cmd.Rerun, app_args=['id', '--resume', '--env', '{"k1": "foobar"}'])
    self.assertEqual(EXPECTED_TASK_RESULT, result[1])