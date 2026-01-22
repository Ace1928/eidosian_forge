import time
import boto
from boto.connection import AWSAuthConnection
from boto.provider import Provider
from boto.exception import SWFResponseError
from boto.swf import exceptions as swf_exceptions
from boto.compat import json
def record_activity_task_heartbeat(self, task_token, details=None):
    """
        Used by activity workers to report to the service that the
        ActivityTask represented by the specified taskToken is still
        making progress. The worker can also (optionally) specify
        details of the progress, for example percent complete, using
        the details parameter. This action can also be used by the
        worker as a mechanism to check if cancellation is being
        requested for the activity task. If a cancellation is being
        attempted for the specified task, then the boolean
        cancelRequested flag returned by the service is set to true.

        :type task_token: string
        :param task_token: The taskToken of the ActivityTask.

        :type details: string
        :param details: If specified, contains details about the
            progress of the task.

        :raises: UnknownResourceFault, SWFOperationNotPermittedError
        """
    return self.json_request('RecordActivityTaskHeartbeat', {'taskToken': task_token, 'details': details})