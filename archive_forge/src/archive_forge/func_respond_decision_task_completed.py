import time
import boto
from boto.connection import AWSAuthConnection
from boto.provider import Provider
from boto.exception import SWFResponseError
from boto.swf import exceptions as swf_exceptions
from boto.compat import json
def respond_decision_task_completed(self, task_token, decisions=None, execution_context=None):
    """
        Used by deciders to tell the service that the DecisionTask
        identified by the taskToken has successfully completed.
        The decisions argument specifies the list of decisions
        made while processing the task.

        :type task_token: string
        :param task_token: The taskToken of the ActivityTask.

        :type decisions: list
        :param decisions: The list of decisions (possibly empty) made by
            the decider while processing this decision task. See the docs
            for the Decision structure for details.

        :type execution_context: string
        :param execution_context: User defined context to add to
            workflow execution.

        :raises: UnknownResourceFault, SWFOperationNotPermittedError
        """
    return self.json_request('RespondDecisionTaskCompleted', {'taskToken': task_token, 'decisions': decisions, 'executionContext': execution_context})