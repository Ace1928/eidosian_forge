from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import json
from googlecloudsdk.api_lib.compute import batch_helper
from googlecloudsdk.api_lib.compute import single_request_helper
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute import waiters
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
import six
from six.moves import zip  # pylint: disable=redefined-builtin
Makes one or more requests to the API.

  Each request can be either a synchronous API call or an asynchronous
  one. For synchronous calls (e.g., get and list), the result from the
  server is yielded immediately. For asynchronous calls (e.g., calls
  that return operations like insert), this function waits until the
  operation reaches the DONE state and fetches the corresponding
  object and yields that object (nothing is yielded for deletions).

  Currently, a heterogeneous set of synchronous calls can be made
  (e.g., get request to fetch a disk and instance), however, the
  asynchronous requests must be homogenous (e.g., they must all be the
  same verb on the same collection). In the future, heterogeneous
  asynchronous requests will be supported. For now, it is up to the
  client to ensure that the asynchronous requests are
  homogenous. Synchronous and asynchronous requests can be mixed.

  Args:
    requests: A list of requests to make. Each element must be a 3-element tuple
      where the first element is the service, the second element is the string
      name of the method on the service, and the last element is a protocol
      buffer representing the request.
    http: An httplib2.Http-like object.
    batch_url: The handler for making batch requests.
    errors: A list for capturing errors. If any response contains an error, it
      is added to this list.
    project_override: The override project for the returned operation to poll
      from.
    progress_tracker: progress tracker to be ticked while waiting for operations
      to finish.
    no_followup: If True, do not followup operation with a GET request.
    always_return_operation: If True, return operation object even if operation
      fails.
    followup_overrides: A list of new resource names to GET once the operation
      finishes. Generally used in renaming calls.
    log_result: Whether the Operation Waiter should print the result in past
      tense of each request.
    log_warnings: Whether warnings for completed operation should be printed.
    timeout: The maximum amount of time, in seconds, to wait for the operations
      to reach the DONE state.

  Yields:
    A response for each request. For deletion requests, no corresponding
    responses are returned.
  