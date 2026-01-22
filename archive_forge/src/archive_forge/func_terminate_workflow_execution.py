import time
import boto
from boto.connection import AWSAuthConnection
from boto.provider import Provider
from boto.exception import SWFResponseError
from boto.swf import exceptions as swf_exceptions
from boto.compat import json
def terminate_workflow_execution(self, domain, workflow_id, child_policy=None, details=None, reason=None, run_id=None):
    """
        Records a WorkflowExecutionTerminated event and forces closure
        of the workflow execution identified by the given domain,
        runId, and workflowId. The child policy, registered with the
        workflow type or specified when starting this execution, is
        applied to any open child workflow executions of this workflow
        execution.

        :type domain: string
        :param domain: The domain of the workflow execution to terminate.

        :type workflow_id: string
        :param workflow_id: The workflowId of the workflow execution
            to terminate.

        :type child_policy: string
        :param child_policy: If set, specifies the policy to use for
            the child workflow executions of the workflow execution being
            terminated. This policy overrides the child policy specified
            for the workflow execution at registration time or when
            starting the execution. The supported child policies are:

            * TERMINATE: the child executions will be terminated.

            * REQUEST_CANCEL: a request to cancel will be attempted
              for each child execution by recording a
              WorkflowExecutionCancelRequested event in its
              history. It is up to the decider to take appropriate
              actions when it receives an execution history with this
              event.

            * ABANDON: no action will be taken. The child executions
              will continue to run.

        :type details: string
        :param details: Optional details for terminating the
            workflow execution.

        :type reason: string
        :param reason: An optional descriptive reason for terminating
            the workflow execution.

        :type run_id: string
        :param run_id: The runId of the workflow execution to terminate.

        :raises: UnknownResourceFault, SWFOperationNotPermittedError
        """
    return self.json_request('TerminateWorkflowExecution', {'domain': domain, 'workflowId': workflow_id, 'childPolicy': child_policy, 'details': details, 'reason': reason, 'runId': run_id})