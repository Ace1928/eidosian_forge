import time
import boto
from boto.connection import AWSAuthConnection
from boto.provider import Provider
from boto.exception import SWFResponseError
from boto.swf import exceptions as swf_exceptions
from boto.compat import json
def poll_for_activity_task(self, domain, task_list, identity=None):
    """
        Used by workers to get an ActivityTask from the specified
        activity taskList. This initiates a long poll, where the
        service holds the HTTP connection open and responds as soon as
        a task becomes available. The maximum time the service holds
        on to the request before responding is 60 seconds. If no task
        is available within 60 seconds, the poll will return an empty
        result. An empty result, in this context, means that an
        ActivityTask is returned, but that the value of taskToken is
        an empty string. If a task is returned, the worker should use
        its type to identify and process it correctly.

        :type domain: string
        :param domain: The name of the domain that contains the task
            lists being polled.

        :type task_list: string
        :param task_list: Specifies the task list to poll for activity tasks.

        :type identity: string
        :param identity: Identity of the worker making the request, which
            is recorded in the ActivityTaskStarted event in the workflow
            history. This enables diagnostic tracing when problems arise.
            The form of this identity is user defined.

        :raises: UnknownResourceFault, SWFOperationNotPermittedError
        """
    return self.json_request('PollForActivityTask', {'domain': domain, 'taskList': {'name': task_list}, 'identity': identity})