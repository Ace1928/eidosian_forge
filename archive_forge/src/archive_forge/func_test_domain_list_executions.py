import boto.swf.layer2
from boto.swf.layer2 import Domain, ActivityType, WorkflowType, WorkflowExecution
from tests.unit import unittest
from mock import Mock
def test_domain_list_executions(self):
    self.domain._swf.list_open_workflow_executions.return_value = {'executionInfos': [{'cancelRequested': False, 'execution': {'runId': '12OeDTyoD27TDaafViz/QIlCHrYzspZmDgj0coIfjm868=', 'workflowId': 'ProcessFile-1.0-1378933928'}, 'executionStatus': 'OPEN', 'startTimestamp': 1378933928.676, 'workflowType': {'name': 'ProcessFile', 'version': '1.0'}}, {'cancelRequested': False, 'execution': {'runId': '12GwBkx4hH6t2yaIh8LYxy5HyCM6HcyhDKePJCg0/ciJk=', 'workflowId': 'ProcessFile-1.0-1378933927'}, 'executionStatus': 'OPEN', 'startTimestamp': 1378933927.919, 'workflowType': {'name': 'ProcessFile', 'version': '1.0'}}, {'cancelRequested': False, 'execution': {'runId': '12oRG3vEWrQ7oYBV+Bqi33Fht+ZRCYTt+tOdn5kLVcwKI=', 'workflowId': 'ProcessFile-1.0-1378933926'}, 'executionStatus': 'OPEN', 'startTimestamp': 1378933927.04, 'workflowType': {'name': 'ProcessFile', 'version': '1.0'}}, {'cancelRequested': False, 'execution': {'runId': '12qrdcpYmad2cjnqJcM4Njm3qrCGvmRFR1wwQEt+a2ako=', 'workflowId': 'ProcessFile-1.0-1378933874'}, 'executionStatus': 'OPEN', 'startTimestamp': 1378933874.956, 'workflowType': {'name': 'ProcessFile', 'version': '1.0'}}]}
    executions = self.domain.executions()
    self.assertEquals(4, len(executions))
    for wf_execution in executions:
        self.assertIsInstance(wf_execution, WorkflowExecution)
        self.assertEquals(self.domain.aws_access_key_id, wf_execution.aws_access_key_id)
        self.assertEquals(self.domain.aws_secret_access_key, wf_execution.aws_secret_access_key)
        self.assertEquals(self.domain.name, wf_execution.domain)
        self.assertEquals(self.domain.region, wf_execution.region)