from unittest import mock
from oslo_serialization import jsonutils
from mistralclient.api.v2.executions import Execution
from mistralclient.api.v2 import tasks
from mistralclient.commands.v2 import tasks as task_cmd
from mistralclient.tests.unit import base
def test_list_with_workflow_execution(self):
    self.client.tasks.list.return_value = [TASK]
    result = self.call(task_cmd.List, app_args=['workflow_execution'])
    self.assertEqual([EXPECTED_TASK_RESULT], result[1])