from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.transfer import jobs_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.transfer import name_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import retry
from googlecloudsdk.core.util import scaled_integer
def block_until_done(job_name=None, operation_name=None):
    """Does not return until API responds that operation is done.

  Args:
    job_name (str|None): If provided, poll job's last operation.
    operation_name (str|None): Poll this operation name.

  Raises:
    ValueError: One of job_name or operation_name must be provided.
  """
    polling_operation_name = _get_operation_to_poll(job_name, operation_name)
    with progress_tracker.ProgressTracker(message='Waiting for operation to complete'):
        retry.Retryer().RetryOnResult(api_get, args=[polling_operation_name], should_retry_if=_is_operation_in_progress, sleep_ms=properties.VALUES.transfer.no_async_polling_interval_ms.GetInt())