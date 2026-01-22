from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import resources
def wait_then_yield_appliance(operation, verb, client=None):
    """Blocks execution until an operation completes and returns an appliance.

  Args:
    operation (messages.Operation): The operation to wait on.
    verb (str): The verb to use in messages, such as "create".
    client (apitools.base.py.base_api.BaseApiService|None): API client for
        loading the results and operations clients.

  Returns:
    poller.GetResult(operation).
  """
    verb += ' appliance'
    return _wait_for_operation(operation, verb, 'appliance', client=client)