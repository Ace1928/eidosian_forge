import time
import boto
from boto.connection import AWSAuthConnection
from boto.provider import Provider
from boto.exception import SWFResponseError
from boto.swf import exceptions as swf_exceptions
from boto.compat import json
def poll_for_decision_task(self, domain, task_list, identity=None, maximum_page_size=None, next_page_token=None, reverse_order=None):
    """
        Used by deciders to get a DecisionTask from the specified
        decision taskList. A decision task may be returned for any
        open workflow execution that is using the specified task
        list. The task includes a paginated view of the history of the
        workflow execution. The decider should use the workflow type
        and the history to determine how to properly handle the task.

        :type domain: string
        :param domain: The name of the domain containing the task
            lists to poll.

        :type task_list: string
        :param task_list: Specifies the task list to poll for decision tasks.

        :type identity: string
        :param identity: Identity of the decider making the request,
            which is recorded in the DecisionTaskStarted event in the
            workflow history. This enables diagnostic tracing when
            problems arise. The form of this identity is user defined.

        :type maximum_page_size: integer :param maximum_page_size: The
            maximum number of history events returned in each page. The
            default is 100, but the caller can override this value to a
            page size smaller than the default. You cannot specify a page
            size greater than 100.

        :type next_page_token: string
        :param next_page_token: If on a previous call to this method a
            NextPageToken was returned, the results are being paginated.
            To get the next page of results, repeat the call with the
            returned token and all other arguments unchanged.

        :type reverse_order: boolean
        :param reverse_order: When set to true, returns the events in
            reverse order. By default the results are returned in
            ascending order of the eventTimestamp of the events.

        :raises: UnknownResourceFault, SWFOperationNotPermittedError
        """
    return self.json_request('PollForDecisionTask', {'domain': domain, 'taskList': {'name': task_list}, 'identity': identity, 'maximumPageSize': maximum_page_size, 'nextPageToken': next_page_token, 'reverseOrder': reverse_order})