from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ai.tensorboard_time_series import client
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import resources
Parse operation relative resource name to the operation reference object.

  Args:
    operation_name: The operation resource name

  Returns:
    The operation reference object
  