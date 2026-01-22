from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.service_extensions import util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import resources
import six
Waits for the opration to complete and returns the result of the operation.

    Args:
      operation_ref: A Resource describing the Operation.
      message: The message to display to the user while they wait.

    Returns:
      result of result_service.Get request for the provided operation.
    