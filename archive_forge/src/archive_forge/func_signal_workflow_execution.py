import time
import boto
from boto.connection import AWSAuthConnection
from boto.provider import Provider
from boto.exception import SWFResponseError
from boto.swf import exceptions as swf_exceptions
from boto.compat import json
def signal_workflow_execution(self, domain, signal_name, workflow_id, input=None, run_id=None):
    """
        Records a WorkflowExecutionSignaled event in the workflow
        execution history and creates a decision task for the workflow
        execution identified by the given domain, workflowId and
        runId. The event is recorded with the specified user defined
        signalName and input (if provided).

        :type domain: string
        :param domain: The name of the domain containing the workflow
            execution to signal.

        :type signal_name: string
        :param signal_name: The name of the signal. This name must be
            meaningful to the target workflow.

        :type workflow_id: string
        :param workflow_id: The workflowId of the workflow execution
            to signal.

        :type input: string
        :param input: Data to attach to the WorkflowExecutionSignaled
            event in the target workflow execution's history.

        :type run_id: string
        :param run_id: The runId of the workflow execution to signal.

        :raises: UnknownResourceFault, SWFOperationNotPermittedError
        """
    return self.json_request('SignalWorkflowExecution', {'domain': domain, 'signalName': signal_name, 'workflowId': workflow_id, 'input': input, 'runId': run_id})