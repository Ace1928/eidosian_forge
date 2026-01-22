import pkg_resources as pkg
from unittest import mock
from oslo_serialization import jsonutils
from mistralclient.api.v2 import executions
from mistralclient.commands.v2 import executions as execution_cmd
from mistralclient.tests.unit import base
def test_update_state(self):
    states = ['RUNNING', 'SUCCESS', 'PAUSED', 'ERROR', 'CANCELLED']
    for state in states:
        self.client.executions.update.return_value = executions.Execution(mock, {'id': '123', 'workflow_id': '123e4567-e89b-12d3-a456-426655440000', 'workflow_name': 'some', 'workflow_namespace': '', 'root_execution_id': '', 'description': '', 'state': state, 'state_info': None, 'created_at': '2020-02-07 08:10:32', 'updated_at': '2020-02-07 08:10:41', 'task_execution_id': None})
        ex_result = list(EX_RESULT)
        ex_result[7] = state
        del ex_result[11]
        ex_result = tuple(ex_result)
        result = self.call(execution_cmd.Update, app_args=['id', '-s', state])
        result_ex = list(result[1])
        del result_ex[11]
        result_ex = tuple(result_ex)
        self.assertEqual(ex_result, result_ex)